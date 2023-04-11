import numpy as np
import matplotlib.pyplot as plt

def load_data(gt_path, pred_path):
    gt_data = np.loadtxt(gt_path)
    # Inverse the ground truth data because it was given wrongly
    gt_data = 1 - gt_data
    detector_data = np.loadtxt(pred_path)
    data = dict([('gt_data', gt_data), ('detector_data', detector_data)])

    return data

def baseline_detector(gt_data):
    baseline_preds = np.zeros(gt_data.shape)
    #baseline_preds = np.random.choice([0, 1.], size=gt_data.shape)
    return baseline_preds

# Main
detector_data_path = r"C:\Users\oksan\anaconda3\envs\dataml_200\ex4\ROC_data\detector_output.dat"
ground_truth_path = r"C:\Users\oksan\anaconda3\envs\dataml_200\ex4\ROC_data\detector_groundtruth.dat"

data = load_data(ground_truth_path, detector_data_path)
gt_data = data['gt_data']
detector_data = data['detector_data']
baseline_data = baseline_detector(gt_data)

# If the data is normally distributed thresholds can be defined as followed
# (This data didn't seem to be)
"""
det_data_mean = np.mean(detector_data)
det_data_var = np.var(detector_data)
print(det_data_mean)
print(det_data_var)

thresholds = [det_data_mean - 4 * det_data_var,
              det_data_mean - 3 * det_data_var,
              det_data_mean - 2 * det_data_var,
              det_data_mean - det_data_var,
              det_data_mean,
              det_data_mean + det_data_var,
              det_data_mean + 2 * det_data_var,
              det_data_mean + 3 * det_data_var,
              det_data_mean + 4 * det_data_var]
print(thresholds)
print()
print(detector_data)
print()
"""

# Number of thresholds
n = 30
thresholds = np.linspace(0, np.max(detector_data) + 0.03, n)

# True positive rates for different thresholds
true_positive_rate = []
baseline_tpr = []
# False positive rates for different thresholds
false_positive_rate = []
baseline_fpr = []

for threshold in thresholds:
    # True positives
    tp = np.sum((detector_data >= threshold) & (gt_data == 1))
    # False negatives
    fn = np.sum((detector_data < threshold) & (gt_data == 1))
    # False positives
    fp = np.sum((detector_data >= threshold) & (gt_data == 0))
    # True negatives
    tn = np.sum((detector_data < threshold) & (gt_data == 0))

    true_pos_rate = tp/(tp+fn)
    false_pos_rate = fp/(fp+tn)

    true_positive_rate.append(true_pos_rate)
    false_positive_rate.append(false_pos_rate)

    # Same for baseline
    # True positives
    tp_baseline = np.sum((baseline_data >= threshold) & (gt_data == 1))
    # False negatives
    fn_baseline = np.sum((baseline_data < threshold) & (gt_data == 1))
    # False positives
    fp_baseline = np.sum((baseline_data >= threshold) & (gt_data == 0))
    # True negatives
    tn_baseline = np.sum((baseline_data < threshold) & (gt_data == 0))

    true_pos_rate_basel = tp_baseline / (tp_baseline + fn_baseline)
    false_pos_rate_basel = fp_baseline / (fp_baseline + tn_baseline)

    baseline_tpr.append(true_pos_rate_basel)
    baseline_fpr.append(false_pos_rate_basel)

"""
print(true_positive_rate)
print()
print(false_positive_rate)
"""
figure1, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7))
ax1.plot(thresholds, true_positive_rate)
ax1.set_title("True positive rates against different thresholds")
ax1.grid()
ax2.plot(thresholds, false_positive_rate)
ax2.set_title("False positive rates against different thresholds")
ax2.grid()
plt.show()

figure2, ax= plt.subplots()
ax.plot(false_positive_rate, true_positive_rate, label="Detector", c='b')
ax.plot(baseline_fpr, baseline_tpr, label="Baseline", c='r')
ax.set_title("ROC-analysis")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.grid()
ax.legend()
plt.show()