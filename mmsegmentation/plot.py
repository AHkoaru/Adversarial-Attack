import matplotlib.pyplot as plt
import numpy as np
from os import makedirs

plot_data_full = {
    "ADE20k": {
        "PSPNet": {
            "queries": [1000, 2000, 3000, 4000, 5000],
            "gt_adv_miou": [0.262, 0.194, 0.162, 0.138, 0.123],
            "pred_adv_miou": [0.3611322581850722, 0.26177308694739887, 0.2017912039826435, 0.1675770154592053, 0.14612747957778044],
            "ratio": [0.009603609432260322, 0.018960770818186694, 0.028045821767928296, 0.03695414180338136, 0.045642790159908214]
        },
        "DeepLabV3": {
            "queries": [1000, 2000, 3000, 4000, 5000],
            "gt_adv_miou": [0.235, 0.174, 0.138, 0.111, 0.092],
            "pred_adv_miou": [0.33497847807536046, 0.22962821652843846, 0.17586247216195547, 0.14347005417031253, 0.11600231953608957],
            "ratio": [0.009656551191278466, 0.01896877636473315, 0.028107953150644804, 0.037009739137123604, 0.045659033520931695]
        },
        "SegFormer": {
            "queries": [1000, 2000, 3000, 4000, 5000],
            "gt_adv_miou": [0.350, 0.327, 0.306, 0.269, 0.249],
            "pred_adv_miou": [0.5279842768133136, 0.43670121135417383, 0.3935045659344523, 0.33618066746303105, 0.30239511730380164],
            "ratio": [0.00947281600656262, 0.01890742437399472, 0.028270759610349225, 0.03759823585348281, 0.04681970450699833]
        },
        "Mask2Former": {
            "queries": [1000, 2000, 3000, 4000, 5000],
            "gt_adv_miou": [0.438, 0.430, 0.430, 0.427, 0.425],
            "pred_adv_miou": [0.7396711048901429, 0.6707433857856647, 0.6660252896804582, 0.6590448610908705, 0.6593477364417897],
            "ratio": [0.005963396491947457, 0.010007729235309395, 0.013167551523742358, 0.016021184047646148, 0.018331892531049526]
        }
    },
    "Cityscapes": {
        "PSPNet": {
            "queries": [1000, 2000, 3000, 4000, 5000],
            "gt_adv_miou": [0.616, 0.494, 0.433, 0.379, 0.339],
            "pred_adv_miou": [0.675243561722005, 0.5265308806781325, 0.45772982496740305, 0.3987139147713414, 0.35357276814915445],
            "ratio": [0.00945481300354004, 0.018795080184936523, 0.02778292655944824, 0.036361827850341796, 0.044485249519348145]
        },
        "DeepLabV3": {
            "queries": [1000, 2000, 3000, 4000, 5000],
            "gt_adv_miou": [0.660, 0.554, 0.484, 0.440, 0.403],
            "pred_adv_miou": [0.7139617243483575, 0.5838727479267661, 0.5047410619733249, 0.45683128644100135, 0.4168833409801277],
            "ratio": [0.009468827247619629, 0.018345637321472166, 0.026727867126464844, 0.03460874080657959, 0.042194852828979494]
        },
        "SegFormer": {
            "queries": [1000, 2000, 3000, 4000, 5000],
            "gt_adv_miou": [0.726, 0.662, 0.620, 0.575, 0.524],
            "pred_adv_miou": [0.7904623787088759, 0.704688636441675, 0.6412360279096467, 0.5811767932139923, 0.5249776838790818],
            "ratio": [0.009706521034240722, 0.019241209030151366, 0.028627471923828127, 0.03786156177520752, 0.04694320678710937]
        },
        "Mask2Former": {
            "queries": [1000, 2000, 3000, 4000, 5000],
            "gt_adv_miou": [0.777, 0.763, 0.756, 0.750, 0.746],
            "pred_adv_miou": [0.8260856923099071, 0.7781071866083009, 0.7625136016334784, 0.7519780012985208, 0.7387264611476947],
            "ratio": [0.0074944162368774415, 0.013424930572509765, 0.01873380184173584, 0.0237508487701416, 0.028405537605285646]
        }
    }
}

model_order = ["PSPNet", "DeepLabV3", "SegFormer", "Mask2Former"]
plt.style.use('seaborn-v0_8-colorblind')
style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
line_markers_on_line = ['^', 's', 'P', 'X']

def create_plot(dataset_name, models_data, miou_key, miou_label_name):
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(f"{dataset_name}: {miou_label_name} (Line) & Pixel Ratio (Circle Size) vs. Queries (Attack @ 5%)", fontsize=16, y=0.98)

    ax.set_xlabel("Number of Queries", fontsize=14)
    ax.set_ylabel(miou_label_name, fontsize=14)
    ax.set_xticks([1000, 2000, 3000, 4000, 5000])
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle=':', alpha=0.6, which='major')

    ax.set_ylim(0, 1)

    # all_mious = []
    # all_ratios = []
    # for model_name_inner in model_order:
    #     if model_name_inner in models_data:
    #         all_mious.extend(models_data[model_name_inner][miou_key])
    #         all_ratios.extend(models_data[model_name_inner]["ratio"])
    
    # min_miou_val = min(all_mious) if all_mious else 0
    # max_miou_val = max(all_mious) if all_mious else 1
    
    # y_bottom_padding = 0.05 * (max_miou_val - (min_miou_val if min_miou_val > 0 else 0))
    # y_top_padding = 0.05 * (max_miou_val - (min_miou_val if min_miou_val > 0 else 0))
    # y_bottom = (min_miou_val if min_miou_val <= 0 else min_miou_val) - y_bottom_padding
    # y_top = max_miou_val + y_top_padding
    # ax.set_ylim(max(0, y_bottom) if miou_key == "gt_adv_miou" or miou_key == "pred_adv_miou" else y_bottom, 
    #             min(1, y_top) if miou_key == "gt_adv_miou" or miou_key == "pred_adv_miou" else y_top)


    max_marker_size = 500
    
    def scale_ratio_to_size(ratio_val, min_ratio, max_ratio):
        scaled_value = (ratio_val - min_ratio) / (max_ratio - min_ratio)
        return ((scaled_value + 1)**3 * max_marker_size) 

    legend_handles = []

    for i, model_name in enumerate(model_order):
        if model_name in models_data:
            data = models_data[model_name]
            queries = data["queries"]
            miou_values = data[miou_key]
            ratios = data["ratio"]
            current_color = style_colors[i % len(style_colors)]

            line, = ax.plot(queries, miou_values, color=current_color, linestyle='-', marker=line_markers_on_line[i], markersize=8, linewidth=2.5, alpha=0.9, label=model_name)
            legend_handles.append(line)

            marker_sizes = [scale_ratio_to_size(r, ratios[0], ratios[-1]) for r in ratios]
            
            ax.scatter(queries, miou_values, s=marker_sizes, color=current_color, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=5, marker='o')

    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=len(model_order), fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    makedirs("plots", exist_ok=True)

    plot_filename = f"plots/{dataset_name}_{miou_key}_line_ratio_dynamic_scatter_size.png"
    plt.savefig(plot_filename, dpi=150)
    print(f"Plot saved as {plot_filename}")

# Generate plots for GT-Adv mIoU
for dataset_name, data_for_dataset in plot_data_full.items():
    create_plot(dataset_name, data_for_dataset, "gt_adv_miou", "GT-Adv mIoU")

# Generate plots for Pred-Adv mIoU
for dataset_name, data_for_dataset in plot_data_full.items():
    create_plot(dataset_name, data_for_dataset, "pred_adv_miou", "Pred-Adv mIoU")

plt.close('all')