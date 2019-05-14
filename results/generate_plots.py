from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

# x = np.arange(4)
# money = [1.5e5, 2.5e6, 5.5e6, 2.0e7]

cp_times = {
        "TF_normal_op"      : (0.00045239925384521484, 'black'),
        "TF_normal_nhwc_op" : (0.00046896934509277344, 'gray' ),
        "TF_original_cp_op" : (0.0007028579711914062, 'orange' ),
        "Custom_fused_op"   : (0.0004673004150390625, 'green' ),
        "sequencer_op_nchw" : (0.0004552602767944336, 'blue' ),
        "sequencer_nhwc_op" : (0.00016498565673828125, 'blue' ),
    }

cp_memory = {
        "TF_normal_op"      : (140288.0, 'black'),
        "TF_normal_nhwc_op" : (140288.0, 'gray' ),
        "TF_original_cp_op" : (91096.0, 'orange' ),
        "Custom_fused_op"   : (132056.0,'green' ),
        "sequencer_op_nchw" : (93568.0, 'blue' ),
        "sequencer_nhwc_op" : (65536.0,  'blue' ),
    }

dense_cp_times = {
        "TF_normal_op"      : (0.00037479400634765625, 'black'),
        "TF_rebuild_op"     : (0.0022580623626708984, 'red' ),
        "TF_einsum_op"      : (0.0019397735595703125, 'blue' ),
        "Custom_fused_op"   : (0.0036014318466186523, 'green'),
        }

dense_cp_memory = {
        "TF_normal_op"      : (1081344.0, 'black'),
        "TF_rebuild_op"     : (145950976.0, 'red'),
        "TF_einsum_op"      : (154877952.0, 'blue'),
        "Custom_fused_op"   : (121856.0, 'green'),
        }

rcp_times = {
        "TF_normal_op"      : (0.0005061626434326172, 'black'),
        "TF_einsum_recomp_op" : (0.000836491584777832, 'red' ),
        "Custom_fused_op"   : (0.0009247064590454102, 'green' ),
        "TF_orig_op"        : (0.000947117805480957, 'orange' ),
        "TF_seq_nchw_op"    : (0.000874638557434082, 'blue' ),
        "TF_seq_nhwc_op"    : (0.0009196996688842773, 'blue'),
    }

rcp_memory = {
        "TF_normal_op"      : (140288.0, 'black'),
        "TF_einsum_recomp_op" : (176128.0, 'red' ),
        "Custom_fused_op"   : (132876.0, 'green' ),
        "TF_orig_op"        : (1442892.0,'orange' ),
        "TF_seq_nchw_op"    : (1442188.0,'blue' ),
        "TF_seq_nhwc_op"    : (1442188.0, 'blue'),
    }


# def millions(x, pos):
#     'The two args are the value and tick position'
#     return '$%1.1fM' % (x * 1e-6)
def microseconds(x, pos):
    return '{:.2E}'.format(x * 1e3)

def megabytes(x, pos):
    return '{:.2E}'.format(x * 1e-6)

# formatter = FuncFormatter(millions)
timeformatter = FuncFormatter(microseconds)
sizeformatter = FuncFormatter(megabytes)

times = [cp_times, rcp_times, dense_cp_times]
memory = [cp_memory, rcp_memory, dense_cp_memory]
names = ['CP Conv2d', 'rCP Conv2d', 'CP Dense']
plot_type = ['Absolute.', 'Relative to Normal Operation.']
formats = [timeformatter, sizeformatter]
# lists = [val for pair in zip(times, memory) for val in pair]

for n, name in enumerate(names):
    for i, D in enumerate((times[n], memory[n])):
        for pt, pt_name in enumerate(plot_type):
            fig, ax = plt.subplots()

            vals_colors = list(D.values())
            vals, colors = zip(*vals_colors)
            if pt != 0:
                tmp = [v / vals[0] for v in vals]
                vals = tmp
                pnm = 'relative Execution Time' if i==0 else 'relative Memory Used'
            else:
                pnm = 'Execution Time (s)' if i==0 else 'Memory Used (MB)'
            #     ax.yaxis.set_major_formatter(formats[i])


            for to_log in range(2):

                lgname = name
                if to_log == 1:
                    lgname = 'log ' + name
                    plt.yscale("log")

                title = lgname + ' ' + pnm + ' ' + pt_name
                plt.title(title)
                plt.ylabel(lgname + ' ' + pnm)
                plt.xlabel('Operations')
                plt.bar(range(len(D)), vals, align='edge', color=colors)
                plt.xticks(range(len(D)), list(D.keys()),  rotation=30)
                for a,b in zip(range(len(D)), vals):
                    plt.text(a, b, "{:.2E}".format(b))
                plt.tight_layout()
                # plt.show()
                plt.savefig(title)
