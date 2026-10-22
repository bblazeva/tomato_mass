# Architecture Agent Brief — Tomato Cluster Segmentation + Mass Estimation

## Purpose

This file summarizes the conclusions from the design/debugging discussion and gives concrete implementation guidance for an execution agent that will modify the existing Detectron2-based RGB-D Mask R-CNN project.

The project goal remains:

1. Detect and instance-segment tomato clusters.
2. Estimate the mass of each segmented tomato cluster.
3. Use RGB-D data so the model learns that apparent image size changes with camera distance, while physical cluster mass does not.

The current direction is **not** to abandon Detectron2. The recommendation is to **keep the working segmentation architecture** and replace/augment the current 2D mass head with a geometry-aware point-based mass head.

---

## Current Understanding of the Existing Model

The existing model is a Detectron2 / Mask R-CNN-style architecture with RGB-D input.

Approximate current structure:

```text
RGB-D image
  -> RGBDGeneralizedRCNN
  -> ResNet-50 + FPN backbone
  -> RPN
  -> Box head
  -> Mask head
  -> Custom mass head
  -> log(mass_kg)
```

The current mass head receives ROI-aligned 2D feature tensors, roughly:

```text
ROI features: 256 FPN channels + 1 depth channel
shape: [num_rois, 257, 14, 14]
```

Then the mass head applies conv/deconv/pooling/MLP layers and predicts a scalar log-mass value.

The model has early RGB-D fusion through the 4-channel input and also had late-fusion attempts by concatenating depth to ROI features.

---

## Key Conclusions From Experiments and Analysis

### 1. Segmentation is not the bottleneck

Segmentation performance is strong. The main failure mode is mass regression, not instance segmentation.

Therefore, do not rewrite the whole detection stack first. Keep Detectron2 for localization and masks.

### 2. The existing 2D mass head appears to underuse depth

A critical ablation showed that using real depth in millimeters versus `depth / 1000.0` produced almost no change in mass results.

That is a strong warning sign. If the architecture really depended on metric depth, this should have changed performance dramatically, because the current depth preprocessing expects millimeters.

Likely issue:

```text
The model is fitting train mass labels using RGB/shape/area/identity shortcuts,
while depth is weakly used or ignored.
```

### 3. Dataset analysis suggests depth is useful, but as metric scale correction

Feature analysis showed the following pattern:

```text
area_px vs mass: moderate correlation
mean/median depth alone vs mass: weak correlation
area_m2 / depth-corrected area proxy vs mass: very strong correlation
pseudo-volume proxy vs mass: strong correlation
```

Interpretation:

```text
Depth is useful, but not as an independent predictor.
Depth is useful because it converts apparent pixel size into metric object size.
```

This means the dataset likely contains a learnable physical signal, but the current 2D ROI mass head is not learning it reliably.

### 4. The dataset does capture camera-distance variation

The same or similar mass values appear across different depth ranges. Some mass groups have large within-mass apparent area variation due to distance changes.

This supports the original data collection idea:

```text
same physical tomato cluster
  photographed at multiple camera distances
  same mass label
  different apparent mask area
```

However, this variance is not uniform across all masses/splits.

### 5. Splits are probably fragile

The available summary showed relatively few unique mass values in validation/test compared with training. Validation/test are sparse and distribution-shifted.

Important for future experiments:

- Prefer grouped splits by physical cluster ID if possible.
- Do not randomly split views of the same physical cluster across train/val/test.
- Always report number of unique mass values and mass-bin distribution per split.

### 6. The apple diameter paper was a reasonable starting point, but the transfer is non-trivial

The inspirational paper predicted **apple diameter**, which is a much more directly visible geometric quantity than **tomato cluster mass**.

Major differences:

```text
Apple diameter:
  mostly single fruit
  roughly spherical
  visible target is close to label
  much larger dataset
  high-confidence / visibility filtering helped

Tomato cluster mass:
  multiple fruits per cluster
  occlusion matters
  hidden tomatoes matter
  cluster compactness matters
  visible surface is only a partial cue
  label is more latent
```

Therefore, simply changing the final linear layer into an MLP is not expected to solve the problem.

---

## Main Architectural Recommendation

### Keep Detectron2. Replace or augment only the mass branch.

Recommended final architecture:

```text
RGB-D image
  -> Detectron2 Mask R-CNN
  -> boxes + masks + FPN features
  -> per-instance RGB-D / XYZ point extraction
  -> PointNet-lite geometry mass head
  -> optional fusion with visual ROI feature
  -> log(mass_kg)
```

Call this architecture something like:

```text
RGB-D Mask R-CNN with Masked XYZ PointNet Mass Head
```

or:

```text
Mask R-CNN + Point-based Mass Head
```

This still counts as an end-to-end neural architecture in the useful sense:

```text
RGB-D image -> segmentation -> masked 3D representation -> neural mass estimate
```

The model is not being given hand-crafted CSV features like `area_m2`. Instead, it is given a better physical representation: per-instance metric 3D points.

---

## Important Design Principle

Do **not** feed the full scene point cloud to the mass head.

The mass label is per tomato-cluster instance. Therefore the mass head input should be:

```text
one segmented/cropped point set per tomato cluster instance
```

Not:

```text
one full point cloud for the whole image
```

Correct conceptual flow:

```text
full RGB-D image / full point cloud
  -> Detectron2 finds one tomato cluster
  -> use its mask/box to extract instance-specific points
  -> point mass head predicts mass for that cluster
```

---

## Recommended Development Plan

### Phase 0 — Standalone PointNet-lite sanity check

Before touching Detectron2 internals, build a standalone PyTorch experiment:

```text
GT segmented tomato-cluster point cloud -> PointNet-lite -> mass
```

This answers the key question:

```text
If segmentation is perfect and the model sees the cluster in metric 3D,
can a neural network predict mass?
```

If this fails, integrating PointNet into Detectron2 will probably not help. If it works, then the current 2D ROI mass head is likely the bottleneck.

#### Dataset format for standalone experiment

One sample = one tomato-cluster instance from one image/view.

Suggested metadata:

```text
sample_id
image_id
cluster_id / physical_cluster_id if available
view_id / camera_distance if available
split
mass_g
num_raw_points
point_cloud_path or generated point array
```

Input tensor shape:

```text
[B, N, C]
```

Where:

```text
B = batch size
N = fixed number of points, e.g. 1024
C = point feature dimension
```

Use a fixed point count by sampling:

```text
if raw points > N: random sample N points
if raw points < N: sample with replacement or pad
```

Start with:

```text
N = 1024
```

Later compare:

```text
N = 512, 2048, 4096
```

#### Recommended point features

Start simple:

```text
[X_rel, Y_rel, Z_rel, Z_abs, R, G, B]
```

Where:

```text
X_rel = X - mean(X)
Y_rel = Y - mean(Y)
Z_rel = Z - mean(Z)
Z_abs = original metric depth or absolute camera Z
RGB normalized to [0, 1]
```

Later try normals:

```text
[X_rel, Y_rel, Z_rel, Z_abs, R, G, B, nx, ny, nz]
```

Normals are useful but should not be required in the first experiment.

#### Critical preprocessing rule

Do **not** normalize each cluster to a unit sphere.

Bad for this project:

```text
center object
then divide by max radius / object extent
```

That destroys physical scale, which is the signal needed for mass.

Good:

```text
center object for translation invariance
keep metric scale in meters or millimeters
```

#### Standalone PointNet-lite model

Suggested architecture:

```text
Input: [B, N, C]

shared point MLP:
  C -> 64 -> 128 -> 256

pooling:
  global max pool -> [B, 256]
  global mean pool -> [B, 256]
  concat -> [B, 512]

regression MLP:
  512 -> 256 -> 128 -> 1

Output:
  log_mass_kg
```

Loss:

```text
MSE(pred_log_mass, gt_log_mass)
```

Report metrics in original units:

```text
MAE_g
RMSE_g
MAPE_percent
R2
```

Also report grouped metrics by:

```text
mass bin
distance/depth bin
cluster visibility / valid depth ratio if available
```

#### Acceptance criteria for Phase 0

The PointNet-lite standalone model should be compared against:

1. Current Detectron2 2D mass head.
2. Simple geometry-feature baseline.
3. Area-only baseline.
4. Depth-only baseline.

If PointNet-lite using GT segmented point clouds beats the current mass head clearly, proceed to integration.

---

## Phase 1 — Integrate point mass head into Detectron2

Do not replace the whole Detectron2 model. Add a new mass head option.

Target design:

```text
RGBDGeneralizedRCNN
  -> cache raw RGB and raw depth
  -> ROIHeads
  -> _forward_mass_pointnet(...)
  -> PointMassHead
```

### Required implementation pieces

#### 1. Cache raw depth and optionally raw RGB

The current code likely caches normalized depth for late fusion. Add raw metric depth too.

Pseudo-structure:

```python
self.roi_heads._cached_depth_raw = depth_mm_or_m
self.roi_heads._cached_depth_norm = depth_norm
self.roi_heads._cached_rgb_raw = rgb_tensor
```

Use consistent units. Prefer meters for XYZ:

```text
depth_m = depth_mm / 1000.0
```

#### 2. Create dense XYZ maps from depth

Given camera intrinsics:

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth_m
```

Dense map shape:

```text
[B, 3, H, W]
```

If intrinsics are unavailable, add config placeholders and fail clearly. Approximate intrinsics can be used only as a temporary experiment, not final.

#### 3. Build dense point-feature map

Suggested dense map:

```text
[B, C, H, W]
```

Channels:

```text
X, Y, Z, R, G, B
```

Optionally later:

```text
normal_x, normal_y, normal_z
valid_depth_mask
```

#### 4. ROIAlign dense XYZ/RGB maps

Use ROIAlign / Detectron2 ROIPooler on the dense point-feature map with stride 1.

Example output:

```text
[R, C, 28, 28]
```

Where `R` is number of selected ROI instances.

Then reshape:

```text
[R, C, 28, 28] -> [R, 784, C]
```

This gives a fixed-size organized point set per ROI.

#### 5. Use mask information

For the first integrated version, use GT masks during training and predicted masks during inference.

Options:

```text
training: GT mask crop
inference: predicted mask crop
```

Better later version:

```text
include mask probability as point feature
point_i = [X, Y, Z, R, G, B, mask_prob]
```

This avoids hard point selection and lets the mass head learn soft weighting.

#### 6. Implement PointMassHead

Create a simple PyTorch module, not full PointNet++ first.

Pseudo-interface:

```python
class PointMassHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 256):
        ...

    def forward(self, points):
        # points: [R, N, C]
        # returns: [R]
        ...
```

The module should:

1. Apply shared MLP to each point.
2. Pool over points using max and mean.
3. Predict scalar log mass.

#### 7. Optional visual fusion

After the point-only version works, fuse point geometry embedding with FPN ROI visual features.

Recommended structure:

```text
point_embedding: [R, 512]
visual_embedding: [R, 256 or 1024]
concat -> MLP -> log_mass
```

Reason: tomato mass may depend on geometry plus RGB-visible tomato count, maturity, occlusion, and texture.

---

## Phase 2 — Training strategy inside Detectron2

Use staged training.

### Stage A — frozen segmentation, GT masks/proposals

Goal:

```text
Can the point mass head work inside Detectron2 when localization is perfect/stable?
```

Use:

```text
GT boxes / GT masks / matched positive proposals
```

Train:

```text
only PointMassHead initially
```

Freeze:

```text
backbone, RPN, box head, mask head
```

### Stage B — positive proposals matched to GT

Use Detectron2 proposal matching. For each positive proposal, identify matched GT instance and mass label.

Extract points using either:

```text
GT mask of matched object
```

or:

```text
proposal box crop + GT mask crop
```

This reduces train/inference mismatch less than pure offline point cloud training.

### Stage C — predicted/soft mask training

Use predicted mask probabilities as point features:

```text
[X, Y, Z, R, G, B, mask_prob]
```

This is more end-to-end and lets the mass branch depend on mask quality.

### Stage D — joint fine-tuning

Unfreeze selected layers only after the point head works.

Suggested first unfreeze:

```text
mass head + ROI visual fusion branch
```

Then maybe:

```text
late FPN layers
```

Avoid unfreezing the full network immediately because the previous model already overfit.

---

## Phase 3 — Cross-distance consistency loss

The dataset was intentionally collected with the same physical cluster at multiple camera distances. Use that.

If `physical_cluster_id` is available, add a consistency loss between views of the same cluster:

```text
L_total = L_mass + lambda_consistency * L_consistency
```

Where:

```text
L_consistency = mean(|pred_i - pred_j|)
```

or in log space:

```text
L_consistency = mean((log_pred_i - log_pred_j)^2)
```

For same physical cluster across distances:

```text
pred_mass(cluster A at 0.3m) ~= pred_mass(cluster A at 0.5m)
pred_mass(cluster A at 0.5m) ~= pred_mass(cluster A at 0.7m)
pred_mass(cluster A at 0.7m) ~= pred_mass(cluster A at 0.9m)
```

This directly encodes the intended physical invariance:

```text
apparent image size may change, mass must not
```

Start with a small weight:

```text
lambda_consistency = 0.05 or 0.1
```

Tune only after the base point head is stable.

---

## Evaluation and Ablation Requirements

Do not evaluate only final MAPE. The goal is to verify depth/geometry use.

Required ablations:

```text
RGB only
Depth/XYZ only
RGB + XYZ
RGB + XYZ + mask_prob
RGB + XYZ + normals
Point-only mass head
Point + visual ROI fusion mass head
GT masks vs predicted masks
real depth vs depth/1000
real depth vs shuffled depth
real RGB vs RGB dropout/blank RGB
```

Important metrics:

```text
MAE_g
RMSE_g
MAPE_percent
R2
mass loss in log space
prediction variance across distances for same physical cluster
performance by mass bin
performance by depth/distance bin
performance by valid_depth_ratio
```

Critical depth-use checks:

```text
If real depth vs depth/1000 gives similar results, the new model is still not using depth.
If real depth vs shuffled depth gives similar results, the new model is still not using geometry properly.
If same-cluster predictions vary strongly across camera distance, invariance is not learned.
```

---

## Research Direction Notes

The papers discussed support the proposed direction but should not be copied wholesale.

### Apple diameter multitask paper

Useful as historical baseline:

```text
Mask R-CNN + RGB-D + regression head can predict apple diameter.
```

But tomato cluster mass is harder than apple diameter.

### Grape Mask R-CNN + PointNet++ paper

Very relevant pipeline idea:

```text
RGB-D -> Mask R-CNN ROI -> point cloud with XYZ/RGB/normals -> PointNet++
```

This supports using a 2D detector to reduce the point-cloud search space.

### Pomegranate F-PointNet paper

Relevant because it combines:

```text
RGB-D Mask R-CNN -> frustum point cloud -> PointNet-like 3D processing
```

Useful warning:

```text
Point cloud sparsity and multiple objects inside one frustum can cause failures.
```

For tomato clusters, track number of points and avoid assuming every crop contains a clean single object unless using masks.

### Strawberry organized point cloud CNN paper

Relevant because it uses organized point clouds from depth cameras and shows that XYZ/normals/color can be processed efficiently.

Useful warning:

```text
Agricultural point clouds can be noisy, reflective, sparse, and sensor-dependent.
```

### FF3D full 3D detector

Interesting but not recommended as first implementation.

Reason:

```text
It is a full 3D detector and would require a much larger rewrite.
Your 2D instance segmentation already works.
```

---

## Suggested File/Module Changes

Actual paths may differ. Search the codebase for these concepts.

### Likely existing areas

```text
RGBDGeneralizedRCNN / meta_arch
custom ROIHeads
mass head module
dataset mapper / loader
mass evaluator
config defaults
training script
```

### New modules to add

```text
point_mass_head.py
point_cloud_utils.py
standalone_point_mass_dataset.py
train_point_mass_standalone.py
evaluate_point_mass.py
```

### Config additions

Add options like:

```yaml
MODEL:
  MASS_HEAD:
    TYPE: "pointnet_lite"   # options: conv2d, pointnet_lite, pointnet_fusion
    POINTS_PER_INSTANCE: 1024
    POINT_FEATURES: ["xyz_rel", "z_abs", "rgb", "mask_prob"]
    ROI_RESOLUTION: 28
    USE_NORMALS: false
    USE_VISUAL_FUSION: false
    FREEZE_DETECTION_FOR_MASS: true
    CONSISTENCY_LOSS_WEIGHT: 0.0

INPUT:
  DEPTH_UNITS: "mm"
  CAMERA_INTRINSICS:
    FX: null
    FY: null
    CX: null
    CY: null
```

---

## Pseudocode — Standalone Dataset

```python
class SegmentedPointMassDataset(torch.utils.data.Dataset):
    def __init__(self, samples, n_points=1024, use_rgb=True, use_normals=False):
        self.samples = samples
        self.n_points = n_points
        self.use_rgb = use_rgb
        self.use_normals = use_normals

    def __getitem__(self, idx):
        sample = self.samples[idx]
        points_xyz = load_xyz(sample.point_cloud_path)  # [M, 3], meters
        rgb = load_rgb_for_points(...)                  # [M, 3], optional

        indices = sample_or_pad_indices(len(points_xyz), self.n_points)
        xyz = points_xyz[indices]
        rgb = rgb[indices]

        centroid = xyz.mean(axis=0, keepdims=True)
        xyz_rel = xyz - centroid
        z_abs = xyz[:, 2:3]

        feats = [xyz_rel, z_abs, rgb]
        x = np.concatenate(feats, axis=1).astype("float32")

        y = np.log(sample.mass_kg).astype("float32")
        return torch.from_numpy(x), torch.tensor(y)
```

---

## Pseudocode — PointNet-lite Mass Head

```python
class PointMassHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
        )
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, points):
        # points: [B, N, C]
        point_features = self.point_mlp(points)     # [B, N, 256]
        max_pool = point_features.max(dim=1).values # [B, 256]
        mean_pool = point_features.mean(dim=1)      # [B, 256]
        global_feat = torch.cat([max_pool, mean_pool], dim=1)
        log_mass = self.regressor(global_feat).squeeze(-1)
        return log_mass
```

---

## Pseudocode — Detectron2 ROI Point Extraction

```python
def build_xyz_map(depth_m, intrinsics):
    # depth_m: [B, 1, H, W]
    # returns xyz: [B, 3, H, W]
    B, _, H, W = depth_m.shape
    device = depth_m.device

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )

    Z = depth_m[:, 0]
    X = (xs[None] - intrinsics.cx) * Z / intrinsics.fx
    Y = (ys[None] - intrinsics.cy) * Z / intrinsics.fy

    xyz = torch.stack([X, Y, Z], dim=1)
    return xyz
```

```python
def roi_points_from_dense_map(dense_map, boxes, roi_pooler):
    # dense_map: [B, C, H, W]
    # boxes: list[Boxes]
    # output: [R, N, C]
    crops = roi_pooler([dense_map], boxes)    # [R, C, 28, 28]
    R, C, Hc, Wc = crops.shape
    points = crops.flatten(2).transpose(1, 2) # [R, 784, C]
    return points
```

---

## Final Recommended Execution Order

1. Build standalone GT segmented point-cloud mass dataset.
2. Train PointNet-lite mass regressor outside Detectron2.
3. Compare against current model and simple baselines.
4. If successful, implement `PointMassHead` in Detectron2.
5. Add raw depth/RGB caching and XYZ map creation.
6. Add ROIAlign-based point extraction.
7. Train with GT masks / frozen detection first.
8. Add predicted mask or soft mask channel.
9. Add optional visual ROI fusion.
10. Add cross-distance consistency loss.
11. Run depth ablations and same-cluster distance-invariance evaluation.

---

## Definition of Success

The architectural change is successful only if it improves both accuracy and physical behavior.

Required success signs:

```text
MAPE/MAE improve over current 2D mass head.
real depth performs better than depth/1000.
real depth performs better than shuffled depth.
predictions for the same cluster are more stable across camera distances.
model does not simply rely on mask pixel area.
```

A model with slightly better MAPE but no depth sensitivity should not be considered a real solution.

---

## Short Summary for the Execution Agent

The current Detectron2 RGB-D Mask R-CNN segments well but the 2D mass head likely ignores depth. Dataset analysis suggests depth-corrected metric geometry is strongly related to mass, but the current ROI conv head does not learn that relationship. Do not rewrite detection. First train a standalone PointNet-lite regressor on GT segmented tomato-cluster point clouds. If it works, replace/augment the Detectron2 mass head with a per-instance masked XYZ/RGB point-based mass head. Use fixed N-point sampling, keep metric scale, avoid unit-sphere normalization, and evaluate depth sensitivity explicitly.
