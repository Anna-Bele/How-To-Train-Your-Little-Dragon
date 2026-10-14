# from tvm.script import ir as I
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis

@I.ir_module
class Module:
    @T.prim_func(s_tir=True)
    def main(astype: T.Buffer((T.int64(1280), T.int64(2592)), "float32"), compute_intermediate: T.Buffer((T.int64(1280), T.int64(2592)), "float16")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        T_softmax_maxelem = T.sblock_alloc_buffer((T.int64(1280),))
        T_softmax_exp = T.sblock_alloc_buffer((T.int64(1280), T.int64(2592)))
        T_softmax_expsum = T.sblock_alloc_buffer((T.int64(1280),))
        T_softmax_norm_intermediate = T.sblock_alloc_buffer((T.int64(1280), T.int64(2592)))
        for i0, k in T.grid(T.int64(1280), T.int64(2592)):
            with T.sblock("T_softmax_maxelem"):
                v_i0, v_k = T.axis.remap("SR", [i0, k])
                T.reads(astype[v_i0, v_k])
                T.writes(T_softmax_maxelem[v_i0])
                with T.init():
                    T_softmax_maxelem[v_i0] = T.float32(-340282346638528859811704183484516925440.0)
                T_softmax_maxelem[v_i0] = T.max(T_softmax_maxelem[v_i0], astype[v_i0, v_k])
        for i0, i1 in T.grid(T.int64(1280), T.int64(2592)):
            with T.sblock("T_softmax_exp"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(astype[v_i0, v_i1], T_softmax_maxelem[v_i0])
                T.writes(T_softmax_exp[v_i0, v_i1])
                T_softmax_exp[v_i0, v_i1] = T.exp(astype[v_i0, v_i1] - T_softmax_maxelem[v_i0])
        for i0, k in T.grid(T.int64(1280), T.int64(2592)):
            with T.sblock("T_softmax_expsum"):
                v_i0, v_k = T.axis.remap("SR", [i0, k])
                T.reads(T_softmax_exp[v_i0, v_k])
                T.writes(T_softmax_expsum[v_i0])
                with T.init():
                    T_softmax_expsum[v_i0] = T.float32(0.0)
                T_softmax_expsum[v_i0] = T_softmax_expsum[v_i0] + T_softmax_exp[v_i0, v_k]
        for i0, i1 in T.grid(T.int64(1280), T.int64(2592)):
            with T.sblock("T_softmax_norm"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_softmax_exp[v_i0, v_i1], T_softmax_expsum[v_i0])
                T.writes(T_softmax_norm_intermediate[v_i0, v_i1])
                T.sblock_attr({"axis": 1})
                T_softmax_norm_intermediate[v_i0, v_i1] = T_softmax_exp[v_i0, v_i1] / T_softmax_expsum[v_i0]
        for i0, i1 in T.grid(T.int64(1280), T.int64(2592)):
            with T.sblock("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_softmax_norm_intermediate[v_i0, v_i1])
                T.writes(compute_intermediate[v_i0, v_i1])
                compute_intermediate[v_i0, v_i1] = T.Cast("float16", T_softmax_norm_intermediate[v_i0, v_i1])