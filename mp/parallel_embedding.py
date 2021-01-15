import os
import numpy
import paddle
from paddle.fluid.incubate.fleet.collective import CollectiveOptimizer, fleet, DistributedStrategy
from paddle.fluid.incubate.fleet.base import role_maker
from paddle.fluid import unique_name
paddle.enable_static()
from paddle.fluid.layer_helper import LayerHelper

def parallel_embedding(
        input,
        size,
        param_attr=None,
        dtype="float32",
        mp_rank=0,
        mp_nranks=1,
        ring_id=0,
        name=None):
    assert len(size) == 2
    assert mp_nranks > 1
    assert size[0] % mp_nranks == 0
    per_rank_emb_size = size[0] // mp_nranks
    per_rank_emb_size += 1  # make the last as the padding index
    origin_input_shape = input.shape
    if len(origin_input_shape) == 2:
        input = paddle.reshape(input, [origin_input_shape[0], origin_input_shape[1], 1])
    else:
        assert origin_input_shape[-1] == 1
    input_shard = paddle.shard_index(input,
                               size[0],
                               mp_nranks,
                               mp_rank,
                               per_rank_emb_size - 1)
    if len(origin_input_shape) == 2:
        input_shard = paddle.reshape(input_shard, origin_input_shape)
    if not name: name = 'emb_rank_%d' % mp_rank
    weight = paddle.create_parameter(shape=[per_rank_emb_size, size[1]],
                                     dtype=dtype,
                                     name=name,
                                     attr=param_attr,
                                     is_bias=False)
    weight.is_distributed = True
    emb = paddle.nn.functional.embedding(x=input_shard,
                                         weight=weight,
                                         padding_idx=per_rank_emb_size - 1)
    paddle.static.default_startup_program().global_block().vars[weight.name].is_distributed = True
    paddle.static.default_main_program().global_block().vars[weight.name].is_distributed = True

    paddle.distributed.all_reduce(emb, group=ring_id)
    return emb


def test_parallel_embedding():
    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)
    numpy.random.seed(2020)
    np_array = numpy.random.rand(10, 12)
    np_data = numpy.random.randint(0, 8, (5,2))

    startup = paddle.fluid.Program()
    main = paddle.fluid.Program()
    with paddle.fluid.program_guard(main, startup):
        data = paddle.static.data(name='data',
                                  shape=[5,2],
                                  dtype="int64")
        if fleet.worker_index() == 0:
            param_attr = paddle.fluid.ParamAttr(
                    initializer=paddle.fluid.initializer.NumpyArrayInitializer(np_array[0:5,:]),
            )
        else:
            param_attr = paddle.fluid.ParamAttr(
                    initializer=paddle.fluid.initializer.NumpyArrayInitializer(np_array[5:10,:]),
            )
        emb_out = parallel_embedding(
            data,
            size=[8, 128],
            param_attr=param_attr,
            mp_rank=fleet.worker_index(),
            mp_nranks=fleet.worker_num(),
            dtype="float32",
            ring_id=0)

        loss = paddle.fluid.layers.reduce_sum(emb_out, 0)
        loss = paddle.mean(loss)

        optimizer = paddle.fluid.optimizer.SGD(0.1)
        dist_strategy = DistributedStrategy()
        dist_strategy.mode = "collective"
        dist_strategy.collective_mode = "grad_allreduce"
        optimizer = fleet.distributed_optimizer(
            optimizer, strategy=dist_strategy)
        optimizer.minimize(loss)

    place = paddle.CUDAPlace(int(os.getenv("FLAGS_selected_gpus", "0")))
    with open("startup", 'w') as f:
        f.writelines(str(startup))
    with open("main_prog", 'w') as f:
        f.writelines(str(main))
    exe = paddle.static.Executor(place)
    exe.run(startup)
    out = exe.run(main,
                  feed={'data': np_data},
                  fetch_list=[emb_out])
    for i in range(np_data.shape[0]):
        for j in range(np_data.shape[1]):
            data = np_data[i][j]
            if data >= 4: data += 1
            assert numpy.allclose(out[0][i][j], np_array[data], atol=1e-08)
            print("passed")
    print("test_parallel_embedding passed.")


if __name__ == "__main__":
    test_parallel_embedding()


