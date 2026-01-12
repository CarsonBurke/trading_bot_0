__global__ void state_passing_fwd_kernel(
    const float* chunk_state,
    const float* exp_a_last,
    const float* initial_state,
    float* state_in,
    float* final_state,
    const int64_t* seq_idx,
    int64_t seq_stride,
    int64_t batch,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t num_chunks,
    int64_t chunk_size,
    int64_t seqlen,
    int64_t has_seq_idx) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * nheads * headdim * d_state;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t n = tmp % d_state;
    tmp /= d_state;
    int64_t p = tmp % headdim;
    tmp /= headdim;
    int64_t h = tmp % nheads;
    int64_t b = tmp / nheads;

    int64_t base = ((b * nheads + h) * headdim + p) * d_state + n;
    float state = initial_state[base];
    for (int64_t c = 0; c < num_chunks; ++c) {
        int64_t cs_idx = ((((b * num_chunks + c) * nheads + h) * headdim + p) * d_state + n);
        if (has_seq_idx && c > 0) {
            int64_t seq_base = b * seq_stride + c * chunk_size;
            int64_t chunk_len = seqlen - c * chunk_size;
            if (chunk_len > chunk_size) {
                chunk_len = chunk_size;
            }
            int64_t seq_last = seq_idx[seq_base + (chunk_len - 1)];
            int64_t seq_prev = seq_idx[seq_base - 1];
            if (seq_last != seq_prev) {
                state = initial_state[base];
            }
        }
        state_in[cs_idx] = state;
        float decay = exp_a_last[(b * nheads + h) * num_chunks + c];
        state = decay * state + chunk_state[cs_idx];
    }
    final_state[base] = state;
}

__global__ void state_passing_bwd_kernel(
    const float* chunk_state,
    const float* state_in,
    const float* exp_a_last,
    const float* dstate_in,
    const float* grad_final_state,
    float* dchunk_state,
    float* ddA,
    float* dstate0,
    int64_t batch,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t seqlen,
    int64_t chunk_size,
    int64_t num_chunks) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * nheads * headdim * d_state;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t n = tmp % d_state;
    tmp /= d_state;
    int64_t p = tmp % headdim;
    tmp /= headdim;
    int64_t h = tmp % nheads;
    int64_t b = tmp / nheads;

    int64_t base = ((b * nheads + h) * headdim + p) * d_state + n;
    float gstate_out = grad_final_state[base];
    for (int64_t c = num_chunks - 1; c >= 0; --c) {
        int64_t last_t = (c + 1) * chunk_size;
        if (last_t > seqlen) {
            last_t = seqlen;
        }
        last_t -= 1;
        int64_t cs_idx = ((((b * num_chunks + c) * nheads + h) * headdim + p) * d_state + n);
        float exp_a = exp_a_last[(b * nheads + h) * num_chunks + c];
        float s_in = state_in[cs_idx];
        atomicAdd(&ddA[(b * nheads + h) * seqlen + last_t], gstate_out * exp_a * s_in);
        dchunk_state[cs_idx] = gstate_out;  // gradient flows directly to chunk_state
        float gstate_in = gstate_out * exp_a + dstate_in[cs_idx];
        gstate_out = gstate_in;
    }
    dstate0[base] = gstate_out;
}

