// SPDX-License-Identifier: AGPL-3.0-only
// Two-host RDMA WRITE ping-pong between GPU-visible host buffers (no GPUDirect).
// Usage: pingpong server <tcp_port> <dev> <gid_idx> <bytes> <iters> <malloc|cudahost>
//        pingpong client <server_ip> <tcp_port> <dev> <gid_idx> <bytes> <iters> <malloc|cudahost>
// Each side registers one buffer. Each message's last 8 bytes carry a sequence
// number; the receiver busy-polls it, then writes back. Reports one-way latency.
#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

struct info { uint32_t qpn, psn, rkey; uint64_t addr; union ibv_gid gid; };

static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static void die(const char *m) { perror(m); exit(1); }

static int cmp(const void *a, const void *b) { double x = *(double *)a, y = *(double *)b; return x < y ? -1 : x > y; }

int main(int argc, char **argv) {
    int server = argc > 1 && !strcmp(argv[1], "server");
    if ((server && argc != 8) || (!server && argc != 9)) { fprintf(stderr, "bad args\n"); return 2; }
    int a = server ? 2 : 3;
    const char *ip = server ? NULL : argv[2];
    int port = atoi(argv[a]), gidx = atoi(argv[a + 2]), iters = atoi(argv[a + 4]);
    const char *devname = argv[a + 1], *memtype = argv[a + 5];
    size_t bytes = strtoull(argv[a + 3], 0, 10);

    int n; struct ibv_device **list = ibv_get_device_list(&n); struct ibv_context *ctx = NULL;
    for (int i = 0; i < n; i++) if (!strcmp(ibv_get_device_name(list[i]), devname)) ctx = ibv_open_device(list[i]);
    if (!ctx) die("open device");
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    struct ibv_cq *cq = ibv_create_cq(ctx, 16, NULL, NULL, 0);
    void *buf;
    if (!strcmp(memtype, "cudahost")) { if (cudaMallocHost(&buf, bytes) != cudaSuccess) die("cudaMallocHost"); }
    else buf = aligned_alloc(4096, (bytes + 4095) / 4096 * 4096);
    memset(buf, 0, bytes);
    struct ibv_mr *mr = ibv_reg_mr(pd, buf, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr) die("reg_mr");
    struct ibv_qp_init_attr qia = { .send_cq = cq, .recv_cq = cq, .qp_type = IBV_QPT_RC,
        .cap = { .max_send_wr = 16, .max_recv_wr = 16, .max_send_sge = 1, .max_recv_sge = 1, .max_inline_data = 0 } };
    struct ibv_qp *qp = ibv_create_qp(pd, &qia);
    if (!qp) die("create_qp");
    struct ibv_qp_attr at = { .qp_state = IBV_QPS_INIT, .port_num = 1, .pkey_index = 0,
        .qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE };
    if (ibv_modify_qp(qp, &at, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) die("init");
    struct info me = { .qpn = qp->qp_num, .psn = 1234, .rkey = mr->rkey, .addr = (uint64_t)buf }, peer;
    if (ibv_query_gid(ctx, 1, gidx, &me.gid)) die("gid");

    int s = socket(AF_INET, SOCK_STREAM, 0), c;
    struct sockaddr_in sa = { .sin_family = AF_INET, .sin_port = htons(port) };
    if (server) {
        int one = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        sa.sin_addr.s_addr = INADDR_ANY;
        if (bind(s, (void *)&sa, sizeof sa) || listen(s, 1)) die("listen");
        c = accept(s, NULL, NULL);
    } else {
        inet_pton(AF_INET, ip, &sa.sin_addr);
        for (int t = 0; connect(s, (void *)&sa, sizeof sa); t++) { if (t > 100) die("connect"); usleep(100000); }
        c = s;
    }
    if (write(c, &me, sizeof me) != sizeof me || read(c, &peer, sizeof peer) != sizeof peer) die("exchange");

    at = (struct ibv_qp_attr){ .qp_state = IBV_QPS_RTR, .path_mtu = IBV_MTU_4096, .dest_qp_num = peer.qpn,
        .rq_psn = peer.psn, .max_dest_rd_atomic = 1, .min_rnr_timer = 12,
        .ah_attr = { .is_global = 1, .port_num = 1, .grh = { .dgid = peer.gid, .sgid_index = gidx, .hop_limit = 64 } } };
    if (ibv_modify_qp(qp, &at, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                      IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) die("rtr");
    at = (struct ibv_qp_attr){ .qp_state = IBV_QPS_RTS, .timeout = 14, .retry_cnt = 7, .rnr_retry = 7,
        .sq_psn = me.psn, .max_rd_atomic = 1 };
    if (ibv_modify_qp(qp, &at, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                      IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) die("rts");
    char sync = 0;
    if (write(c, &sync, 1) != 1 || read(c, &sync, 1) != 1) die("sync");

    volatile uint64_t *flag = (volatile uint64_t *)((char *)buf + bytes - 8);
    struct ibv_sge sge = { .addr = (uint64_t)buf, .length = bytes, .lkey = mr->lkey };
    struct ibv_send_wr wr = { .sg_list = &sge, .num_sge = 1, .opcode = IBV_WR_RDMA_WRITE,
        .send_flags = IBV_SEND_SIGNALED, .wr.rdma = { .remote_addr = peer.addr, .rkey = peer.rkey } }, *bad;
    double *lat = malloc(sizeof(double) * iters);
    struct ibv_wc wc;
    for (int i = 1; i <= iters + 100; i++) {
        double t0 = now_us();
        if (!server) {
            *flag = i;
            if (ibv_post_send(qp, &wr, &bad)) die("post");
            while (*flag != (uint64_t)-(int64_t)i) ;           // wait for the echo
        } else {
            while (*flag != (uint64_t)i) ;                     // wait for the ping
            *flag = (uint64_t)-(int64_t)i;
            if (ibv_post_send(qp, &wr, &bad)) die("post");
        }
        while (ibv_poll_cq(cq, 1, &wc) == 0) ;
        if (wc.status != IBV_WC_SUCCESS) { fprintf(stderr, "wc %s\n", ibv_wc_status_str(wc.status)); return 1; }
        if (!server && i > 100) lat[i - 101] = (now_us() - t0) / 2;
    }
    if (!server) {
        qsort(lat, iters, sizeof *lat, cmp);
        printf("{\"mem\":\"%s\",\"bytes\":%zu,\"one_way_us_p50\":%.2f,\"p10\":%.2f,\"p90\":%.2f,\"p99\":%.2f}\n",
               memtype, bytes, lat[iters / 2], lat[iters / 10], lat[iters * 9 / 10], lat[iters * 99 / 100]);
    }
    return 0;
}
