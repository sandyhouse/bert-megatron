import paddle


def parallel_mlp(input,
                 hidden_size,
                 use_bias=True,
                 gather_output=True,
                 param_attr=None,
                 bias_attr=None,
                 mp_rank=0,
                 mp_nranks=1,
                 dtype="float32",
                 param_name=None,
                 bias_name=None,
                 ring_id=0):
    assert out_size % mp_nranks == 0
    out = column_parallel_linear(input,
                                 hidden_size,
                                 hidden_size*4,
                                 gather_out=False,
                                 mp_rank=mp_rank,
                                 mp_nranks=mp_nranks,
                                 use_bias=False,
                                 ring_id=ring_id)
    out = row_parallel_linear(out,
                              hidden_size*4,
                              hidden_size,
                              gather_out=True,
                              mp_rank=mp_rank,
                              mp_nranks=mp_nranks,
                              use_bias=True,
                              ring_id=ring_id)
    return out



