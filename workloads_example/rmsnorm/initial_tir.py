# from tvm.script import ir as I
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis

@I.ir_module
class Module:
    @T.prim_func(s_tir=True)
    def main(x: T.Buffer((T.int64(1), T.int64(720), T.int64(4096)), "float16"), weight: T.Buffer((T.int64(4096),), "float16"), T_cast: T.Buffer((T.int64(1), T.int64(720), T.int64(4096)), "float16")):
        T.func_attr({"op_pattern": 4, "tirx.noalias": True})
        # with T.sblock("root"):
        T_cast_1 = T.sblock_alloc_buffer((T.int64(1), T.int64(720), T.int64(4096)))
        T_multiply = T.sblock_alloc_buffer((T.int64(1), T.int64(720), T.int64(4096)))
        T_multiply_red = T.sblock_alloc_buffer((T.int64(1), T.int64(720)))
        rsqrt = T.sblock_alloc_buffer((T.int64(1), T.int64(720)))
        T_cast_2 = T.sblock_alloc_buffer((T.int64(4096),))
        T_rms_norm = T.sblock_alloc_buffer((T.int64(1), T.int64(720), T.int64(4096)))
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(720), T.int64(4096)):
            with T.sblock("T_cast"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(x[v_ax0, v_ax1, v_ax2])
                T.writes(T_cast_1[v_ax0, v_ax1, v_ax2])
                T_cast_1[v_ax0, v_ax1, v_ax2] = T.Cast("float32", x[v_ax0, v_ax1, v_ax2])
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(720), T.int64(4096)):
            with T.sblock("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_cast_1[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                T_multiply[v_ax0, v_ax1, v_ax2] = T_cast_1[v_ax0, v_ax1, v_ax2] * T_cast_1[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, k2 in T.grid(T.int64(1), T.int64(720), T.int64(4096)):
            with T.sblock("T_multiply_red"):
                v_ax0, v_ax1, v_k2 = T.axis.remap("SSR", [ax0, ax1, k2])
                T.reads(T_multiply[v_ax0, v_ax1, v_k2])
                T.writes(T_multiply_red[v_ax0, v_ax1])
                with T.init():
                    T_multiply_red[v_ax0, v_ax1] = T.float32(0.0)
                T_multiply_red[v_ax0, v_ax1] = T_multiply_red[v_ax0, v_ax1] + T_multiply[v_ax0, v_ax1, v_k2]
        for ax0, ax1 in T.grid(T.int64(1), T.int64(720)):
            with T.sblock("rsqrt"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(T_multiply_red[v_ax0, v_ax1])
                T.writes(rsqrt[v_ax0, v_ax1])
                rsqrt[v_ax0, v_ax1] = T.rsqrt(T_multiply_red[v_ax0, v_ax1] / T.float32(4096.0) + T.float32(9.9999999999999995e-07))
        for ax0 in range(T.int64(4096)):
            with T.sblock("T_cast_1"):
                v_ax0 = T.axis.spatial(T.int64(4096), ax0)
                T.reads(weight[v_ax0])
                T.writes(T_cast_2[v_ax0])
                T_cast_2[v_ax0] = T.Cast("float32", weight[v_ax0])
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(720), T.int64(4096)):
            with T.sblock("T_rms_norm"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(rsqrt[v_ax0, v_ax1], T_cast_1[v_ax0, v_ax1, v_ax2], T_cast_2[v_ax2])
                T.writes(T_rms_norm[v_ax0, v_ax1, v_ax2])
                T_rms_norm[v_ax0, v_ax1, v_ax2] = rsqrt[v_ax0, v_ax1] * T_cast_1[v_ax0, v_ax1, v_ax2] * T_cast_2[v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(720), T.int64(4096)):
            with T.sblock("T_cast_2"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_rms_norm[v_ax0, v_ax1, v_ax2])
                T.writes(T_cast[v_ax0, v_ax1, v_ax2])
                T_cast[v_ax0, v_ax1, v_ax2] = T.Cast("float16", T_rms_norm[v_ax0, v_ax1, v_ax2])