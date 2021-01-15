import paddle


def parallel_self_attention(input,
                            hidden_size,
                            num_attention_heads,
                            mp_rank,
                            mp_nranks,
                            dtype="float32",
                            ring_id=0):
    assert hidden_size % mp_nranks == 0
    hidden_size_per_part = hidden_size // mp_nranks
    assert hidden_size % num_attention_head == 0
    hidden_size_per_head = hidden_size // num_attention_heads
    assert num_attention_heads % mp_nranks == 0
    num_attention_head_per_part = num_attention_heads // mp_nranks

    query_key_value = column_parallel_linear(input,
                                             hidden_size,
                                             hidden_size*3,
                                             use_bias=False,
                                             gather_out=False,
                                             mp_rank=mp_rank,
                                             mp_nranks=mp_nranks,
                                             dtype=dtype,
                                             ring_id=ring_id)
    # [sq, b, (np * hn * 3)] -> [sq, b, np, 3 * bn]
    new_shape = query_key_value.shape()[:-1] + (
        num_attention_head_per_part, 3 * hidden_size_per_part)
    query_key_value = paddle.reshape(query_key_value, new_shape)

    # [sq, b, np, 3*bn] -> 3 [sq, b, np, bn]
    (query, key, value) = paddle.split(query_key_value, 3)

    # [b, np, sq, sk]
    output_size = (query.shape[1],
                   query.shape[2],
                   query.shape[0],
                   key.shape[0])

    # [sq, b, np, bn] -> [sq, b * np, hn]
    query = paddle.reshape(query, (output_size[2],
                                   output_size[0] * output_size[1],
                                   -1))
    key = paddle.reshape(key, (output_size[3],
                                   output_size[0] * output_size[1],
                                   -1))
    result = paddle.bmm(paddle.transpose(query, [1, 0, 2]),
                        paddle.transpose(query, [1, 2, 0]))
    result = paddle.scale(result, 1.0/norm_factor)
    scores = paddle.reshape(result, output_size)

    # [b, np, sq, hn]
    output_size = (value.shape[1],
                   value.shape[2],
                   query.shape[0],
                   value.shape[3])

    # [sk, b * np, hn]
    value = paddle.reshape(value, [value.shape[0],
                                   output_size[0] * output_size[1],
                                   -1])

    # [b * np, sq, sk]
    attention_probs = paddle.reshape(attention_probs,
                                     [output_size[0] * output_size[1],
                                      output_size[2],
                                      -1])
    
    [b * np, sq, bn]
    context = paddle.bmm(attention_probs,
                         paddle.transpose(value, [1, 0, 2]))

    # [b, np, sq, hn]
    context = paddle.reshape(context, output_size)

    #[b, np, sq, hn] -> [sq, b, np, hn]
    context = paddle.transpose(context, [2, 0, 1, 3])
    
    # [sq, b, np, hn] -> [sq, b, hp]
    new_shape = context.shape[:-2] + (hidden_size_per_part,)
    context = paddle.reshape(context, new_shape)

    # Output: [sq, b, h]
    output, bias = row_parallel_linear()



