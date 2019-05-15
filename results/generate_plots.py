from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

# x = np.arange(4)
# money = [1.5e5, 2.5e6, 5.5e6, 2.0e7]

cp_times = {
        "TNN_original_cp_op" : (0.0007028579711914062, 'orange' ),
        "Custom_fused_op"   : (0.0004673004150390625, 'green' ),
        "TF_sequencer_op_nchw" : (0.0004552602767944336, 'blue' ),
        "TF_sequencer_op_nhwc" : (0.00016498565673828125, 'blue' ),
        "TF_rebuild_op"     : (0.00045228004455566406, 'red' ),
        "TF_Full_op_nchw"   : (0.00045239925384521484, 'black'),
        "TF_Full_op_nhwc"   : (0.00046896934509277344, 'gray' ),
    }

cp_memory = {
        "TNN_original_cp_op" : (91096.0, 'orange' ),
        "Custom_fused_op"   : (132056.0,'green' ),
        "TF_sequencer_op_nchw" : (93568.0, 'blue' ),
        "TF_sequencer_op_nhwc" : (65536.0,  'blue' ),
        "TF_rebuild_op"     : (140288.0, 'red' ),
        "TF_Full_op_nchw"   : (140288.0, 'black'),
        "TF_Full_op_nhwc"   : (140288.0, 'gray' ),
    }

dense_cp_times = {
        "TNN_original_cp_op" : (0.0009917020797729492, 'orange'),
        "Custom_fused_op"   : (0.0035014318466186523, 'green'),
        "TF_sequence_op"    : (0.0010008811950683594, 'blue'),
        "TF_rebuild_op"     : (0.0022580623626708984, 'red' ),
        "TF_einsum_op"      : (0.0019397735595703125, 'pink' ),
        "TF_Full_op"        : (0.00045079400634765625, 'black'),
    }

dense_cp_memory = {
        "TNN_original_cp_op" : (1192448.0, 'orange'),
        "Custom_fused_op"   : (121856.0, 'green'),
        "TF_sequence_op"    : (1192448.0, 'blue'),
        "TF_rebuild_op"     : (145950976.0, 'red'),
        "TF_einsum_op"      : (154877952.0, 'pink'),
        "TF_Full_op"      : (1081344.0, 'black'),
    }

rcp_times = {
        "TNN_orig_op"        : (0.000947117805480957, 'orange' ),
        "Custom_fused_op"   : (0.0009247064590454102, 'green' ),
        "TF_seq_nchw_op"    : (0.001069784164428711, 'blue' ),
        "TF_seq_nhwc_op"    : (0.0011096996688842773, 'blue'),
        "TF_einsum_recomp_op" : (0.000836491584777832, 'red' ),
        "TF_Full_op"      : (0.0005061626434326172, 'black'),
    }

rcp_memory = {
        "TTNN_orig_op"        : (1442892.0,'orange' ),
        "Custom_fused_op"   : (132876.0, 'green' ),
        "TF_seq_nchw_op"    : (1442188.0,'blue' ),
        "TF_seq_nhwc_op"    : (1442188.0, 'blue'),
        "TF_einsum_recomp_op" : (176128.0, 'red' ),
        "TF_Full_op"      : (140288.0, 'black'),
    }

def microseconds(x, pos):
    return '{:.2f}'.format(x * 1e6)

def megabytes(x, pos):
    return '{:.2f}'.format(x * 1e-6)

timeformatter = FuncFormatter(microseconds)
sizeformatter = FuncFormatter(megabytes)

times = [cp_times, rcp_times, dense_cp_times]
memory = [cp_memory, rcp_memory, dense_cp_memory]
names = ['CP Conv2d', 'rCP Conv2d', 'CP Dense']
formats = [timeformatter, sizeformatter]

for n, name in enumerate(names):
    for i, D in enumerate((times[n], memory[n])):
        # for pt, pt_name in enumerate(plot_type):
        fig, ax = plt.subplots()


        vals_colors = list(D.values())
        vals, colors = zip(*vals_colors)

        ratios = [v / vals[0] for v in vals]

        for to_log in range(2):
            lgname = ''
            fmt = formats[i]

            pnm = 'Mean Execution Time' if i==0 else 'Memory Used'
            if to_log == 1:
                lgname = 'log'
                plt.yscale("log")
                unit = r'($s$)' if i==0 else '(B)'
            else:
                unit = r'($\mu s$)' if i==0 else '(MB)'
                ax.yaxis.set_major_formatter(formats[i])

            title = lgname+' '+name+' '+pnm
            plt.title(title)
            plt.ylabel(lgname + ' ' + pnm + ' ' + unit)
            plt.xlabel('Operations')
            plt.bar(range(len(D)), vals, align='edge', color=colors)
            plt.xticks(range(len(D)), list(D.keys()),  rotation=30)
            for j,v,r in zip(range(len(D)), vals, ratios):
                plt.text(j, v, "{:.3f}x".format(r))
            plt.tight_layout()
            # plt.show()
            plt.savefig(title, dpi=240)
