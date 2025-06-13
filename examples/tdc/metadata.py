from losses_and_metrics import (SparseL1, 
                                SparseAUROC,
                                SparseAUPRC,
                                SparseSpearman,
                                SparseMSELoss,
                                SparseCELoss)

metrics_to_metrics_fn = {
    "mae": SparseL1,
    "roc-auc": SparseAUROC,
    "pr-auc": SparseAUPRC,
    "spearman": SparseSpearman,
}

metrics_to_loss_fn = {
    "mae": SparseMSELoss,
    "roc-auc": SparseCELoss,
    "pr-auc": SparseCELoss,
    "spearman": SparseMSELoss,
}

admet_metrics = {
    "caco2_wang": "mae",
    "hia_hou": "roc-auc",
    "pgp_broccatelli": "roc-auc",
    "bioavailability_ma": "roc-auc",
    "lipophilicity_astrazeneca": "mae",
    "solubility_aqsoldb": "mae",
    "bbb_martins": "roc-auc",
    "ppbr_az": "mae",
    "vdss_lombardo": "spearman",
    "cyp2c9_veith": "pr-auc",
    "cyp2d6_veith": "pr-auc",
    "cyp3a4_veith": "pr-auc",
    "cyp2c9_substrate_carbonmangels": "pr-auc",
    "cyp3a4_substrate_carbonmangels": "roc-auc",
    "cyp2d6_substrate_carbonmangels": "pr-auc",
    "half_life_obach": "spearman",
    "clearance_hepatocyte_az": "spearman",
    "clearance_microsome_az": "spearman",
    "ld50_zhu": "mae",
    "herg": "roc-auc",
    "ames": "roc-auc",
    "dili": "roc-auc",
}

more_tasks = {
    # extra datasets - tox
    "skin_reaction": "roc-auc",
    "carcinogens_lagunin": "roc-auc",
    "clintox": "roc-auc",
    # extra datasets - adme
    "hydrationfreeenergy_freesolv": "mae",
    "pampa_ncats":"roc-auc",
    "cyp1a2_veith": "pr-auc",
}

leaderboard = {
    "caco2_wang":                     [0.276, 0.285, 0.287, 0.289, 0.321, 0.330, 0.341, 0.344, 0.393, 0.401, 0.446, 0.502, 0.530, 0.546, 0.599, 0.908],
    "bioavailability_ma":             [0.748, 0.742, 0.730, 0.706, 0.672, 0.671, 0.667, 0.632, 0.613, 0.581, 0.577, 0.566, 0.523],
    "lipophilicity_astrazeneca":      [0.467, 0.470, 0.479, 0.525, 0.535, 0.539, 0.541, 0.547, 0.563, 0.572, 0.574, 0.617, 0.621, 0.701, 0.743],
    "solubility_aqsoldb":             [0.761, 0.776, 0.789, 0.792, 0.827, 0.828, 0.829, 0.907, 0.947, 1.023, 1.026, 1.040, 1.076, 1.203],
    "hia_hou":                        [0.989, 0.988, 0.986, 0.981, 0.978, 0.975, 0.974, 0.972, 0.965, 0.948, 0.943, 0.936, 0.926, 0.869, 0.818, 0.807],
    "pgp_broccatelli":                [0.938, 0.935, 0.930, 0.929, 0.923, 0.918, 0.908, 0.902, 0.895, 0.892, 0.886, 0.880, 0.860, 0.845, 0.818],
    "bbb_martins":                    [0.916, 0.915, 0.913, 0.912, 0.910, 0.908, 0.905, 0.903, 0.901, 0.897, 0.892, 0.889, 0.869, 0.855, 0.842, 0.836, 0.823, 0.821, 0.811, 0.781, 0.725],
    "ppbr_az":                        [7.526, 7.660, 7.788, 7.914, 8.288, 9.185, 9.292, 9.373, 9.445, 9.942, 9.994, 10.075, 10.194, 11.106, 12.848],
    "vdss_lombardo":                  [0.713, 0.707, 0.627, 0.609, 0.582, 0.561, 0.559, 0.493, 0.491, 0.485, 0.457, 0.389, 0.258, 0.241, 0.226],
    "cyp2c9_veith":                   [0.859, 0.839, 0.829, 0.786, 0.783, 0.777, 0.767, 0.754, 0.749, 0.742, 0.739, 0.735, 0.715, 0.713, 0.556, 0.536],
    "cyp2d6_veith":                   [0.790, 0.739, 0.723, 0.721, 0.673, 0.649, 0.646, 0.644, 0.627, 0.616, 0.587, 0.544, 0.358, 0.348],
    "cyp3a4_veith":                   [0.916, 0.904, 0.902, 0.881, 0.876, 0.875, 0.862, 0.851, 0.849, 0.840, 0.829, 0.827, 0.821, 0.696, 0.654],
    "cyp2c9_substrate_carbonmangels": [0.441, 0.437, 0.433, 0.415, 0.400, 0.392, 0.382, 0.381, 0.380, 0.375, 0.367, 0.360, 0.359, 0.347, 0.344, 0.281],
    "cyp2d6_substrate_carbonmangels": [0.736, 0.720, 0.713, 0.704, 0.686, 0.685, 0.677, 0.671, 0.632, 0.617, 0.574, 0.572, 0.498, 0.485, 0.478],
    "cyp3a4_substrate_carbonmangels": [0.662, 0.650, 0.647, 0.640, 0.639, 0.633, 0.630, 0.629, 0.619, 0.609, 0.605, 0.596, 0.590, 0.582, 0.578, 0.576],
    "half_life_obach":                [0.562, 0.557, 0.547, 0.544, 0.438, 0.392, 0.329, 0.265, 0.239, 0.184, 0.177, 0.151, 0.129, 0.085, 0.038],
    "clearance_hepatocyte_az":        [0.498, 0.466, 0.440, 0.439, 0.431, 0.430, 0.424, 0.413, 0.401, 0.382, 0.366, 0.289, 0.272, 0.235],
    "clearance_microsome_az":         [0.630, 0.626, 0.625, 0.599, 0.597, 0.586, 0.585, 0.578, 0.572, 0.555, 0.532, 0.529, 0.518, 0.492, 0.365, 0.252],
    "ld50_zhu":                       [0.552, 0.588, 0.606, 0.621, 0.622, 0.625, 0.633, 0.636, 0.646, 0.649, 0.667, 0.669, 0.675, 0.678, 0.685],
    "herg":                           [0.880, 0.874, 0.871, 0.856, 0.841, 0.840, 0.825, 0.778, 0.756, 0.754, 0.749, 0.738, 0.736, 0.722, 0.721, 0.715],
    "ames":                           [0.871, 0.869, 0.868, 0.850, 0.842, 0.837, 0.823, 0.818, 0.814, 0.794, 0.776, 0.755, 0.716],
    "dili":                           [0.925, 0.919, 0.917, 0.909, 0.899, 0.887, 0.886, 0.875, 0.873, 0.861, 0.859, 0.851, 0.832, 0.792, 0.700]
}                                      #1     #2     #3     #4     #5     #6     #7     #8     #9     #10    #11    #12    #13    #14    #15    #16
