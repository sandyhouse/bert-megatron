import paddle


def column_parallel_linear(
        input,
        in_size,
        out_size,
        use_bias=True,
        gather_out=True,
        mp_rank=0,
        mp_nranks=1,
        dtype="float32",
        param_attr=None,
        bias_attr=None,
        param_name=None,
        bias_name=None,
        ring_id=0):
    assert out_size % mp_nranks == 0
    out_size_per_part = out_size // mp_nranks
    weight = paddle.create_parameter(shape=[in_size, out_size_per_part],
                                     dtype=dtype,
                                     name=param_name,
                                     attr=param_attr,
                                     is_bias=False)
    weight.is_distributed = True
    paddle.static.default_startup_program().global_block().vars[weight.name].is_distributed = True
    paddle.static.default_main_program().global_block().vars[weight.name].is_distributed = True
    if use_bias:
        bias = paddle.create_parameter(shape=[out_size_per_part],
                                       dtype=dtype,
                                       name=bias_name,
                                       attr=param_attr,
                                       is_bias=True)
        bias.is_distributed = True
        paddle.static.default_startup_program().global_block().vars[bias.name].is_distributed = True
        paddle.static.default_main_program().global_block().vars[bias.name].is_distributed = True
    out = paddle.matmul(input, weight)
    if use_bias:
        out = paddle.elementwise_add(out, bias)
    if gather_out:
        output = []
        paddle.distributed.all_gather(output, out, group=ring_id)
        out = paddle.concat(output, axis=len(out.shape)-1)
    return out

