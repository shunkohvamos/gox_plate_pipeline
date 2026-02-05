# Why the BO gradient map looks like horizontal stripes (MPC / y bias)

## Summary

- **Cause**: The GP learns that the objective (log FoG) varies strongly with **y** (= 1 − MPC) and hardly with **x** (= BMA/(BMA+MTAC)), so the predicted surface is almost constant along x → **horizontal bands** in the ternary plot.
- **Root cause**: The **training design** has too little **x-diversity within each y**. For most y-levels there is only **one** x value, so the model cannot learn how the response changes with x.
- **Catalog**: The composition catalog (and which polymers were actually measured) is the source of this design: it has many different **y** values but, for most of them, only one **x** (one BMA/MTAC ratio). So the catalog/experiment design is the main cause.

---

## 1. What are x and y in this project?

In the BO code (e.g. `bo_data.py`, `bo_engine.py`):

- **y** = `frac_BMA + frac_MTAC` = **1 − frac_MPC**  
  → "How much non-MPC" (vertical direction in the ternary plot).
- **x** = `frac_BMA / (frac_BMA + frac_MTAC)` when (BMA+MTAC) > 0  
  → "BMA vs MTAC ratio" **within** a fixed non-MPC amount (horizontal direction in a band).

So:

- **Same y** = same MPC level → one horizontal band.
- **Same y, different x** = same MPC, different BMA/MTAC split.

The GP is trained on 2D inputs **(x, y)** and predicts log FoG.

---

## 2. What does "11 y-levels, 8 of which have only 1 x value" mean?

It means: if you group the **unique design points** (each polymer = one (x, y)) by **y**:

- You get about **10–11 distinct y values** (e.g. 0.2, 0.45, 0.62, 0.7, 0.88, 0.9, 1.0, …).
- For **8 of those y values**, there is **only one x** in the data (only one BMA/MTAC ratio at that MPC level).
- For **only 2 y values** (in the current run: y=0.2 and y=0.7) there are **two x values** (two different BMA/MTAC ratios).

So:

- **"That y has only 1 x"** = at that MPC level we only have one composition in the catalog/measurements (one BMA/MTAC ratio). We have no information to learn "what happens if we change BMA/MTAC at this MPC".
- **"That y has 2 x values"** = at that MPC level we have two compositions (e.g. BMA-rich and MTAC-rich), so the model can learn a bit how the response changes with x.

Example from `bo_training_data__bo_2026-02-05_10-52-46.csv` (unique design by `polymer_id`):

| y (=1−MPC) | n_points | n_x_values | Note |
|------------|----------|------------|------|
| 0.2        | 2        | 2          | PMBTA-1 (x≈0.2), PMBTA-2 (x≈0.8) |
| 0.45       | 1        | 1          | PMBTA-5 only |
| 0.62, 0.63, 0.65, 0.88, 0.89, 0.91, 1.0 | 1 each | 1 each | Only one composition per y |

So **8 y-levels have only 1 x**, **2 y-levels have 2 x**.

---

## 3. Why does the model become "MPC-biased" (y-dependent)?

- The GP fits a 2D surface (x, y) → log FoG.
- Where we have **several y values** but **only one x per y** (for most y), the data tell the model:
  - "When y changes, log FoG changes" → strong **y** effect → **short length scale in y**.
  - "When x changes (at fixed y), we have almost no data" → the model assumes the function is smooth and almost constant in x → **long length scale in x**.

So we get:

- **length_scale_y** small (e.g. ~0.07) → response changes quickly with y (MPC).
- **length_scale_x** large (e.g. ~8) → response almost constant in x (BMA/MTAC ratio).

On the ternary plot, **y is roughly the vertical direction** (MPC from top to bottom). So "constant in x, varying in y" → **horizontal stripes**.

So the map is not "broken": the **learned landscape** is literally band-like because the **training design** is band-like (lots of y variation, little x variation per y).

---

## 4. Is there a problem with the composition catalog?

**Yes.** The issue is **design coverage**, not a single bug:

- The **catalog** (and the set of polymers that were actually run and have FoG) has:
  - Many different **MPC levels** (many y) ✓  
  - For **most** of those MPC levels, **only one** BMA/MTAC ratio (one x) ✗  

So the catalog (and the experiments) do not fill the **(x, y)** space in a balanced way: they fill "many horizontal bands" but each band has only one (or very few) x. That is why the model learns "y matters, x doesn't" and the gradient map looks like horizontal stripes.

**What would help:**

- For **each** (or many) **y** (each MPC level), include **several x** values (several BMA/MTAC ratios), e.g. 2–3 compositions per y.
- Then the GP would see how log FoG changes with **both** y and x, and length_scale_x would not be forced to be huge → smoother, non-striped contours.

In short: the stripes come from **data/catalog design** (too little x-diversity per y), not from a plotting or GP implementation bug.

---

## 5. Is the cause "forcing ternary into 2D (x, y)"?

**No—and (x, y) is optimal.** The ternary composition has **2 degrees of freedom** (three fractions that sum to 1). The (x, y) parameterization is the **optimal** 2D representation:

- **x** = frac_BMA / (frac_BMA + frac_MTAC) → BMA/MTAC ratio (one independent degree of freedom)
- **y** = frac_BMA + frac_MTAC = 1 − frac_MPC → non-MPC amount (another independent degree of freedom)

This directly represents the two independent degrees of freedom with **no redundancy**. So "3D into 2D" is not a problem—it's the correct dimensionality. The cause is **design sparsity**: at most y-levels we only have one x, so the GP cannot learn how the response varies with x.

---

## 6. Can we do BO in "3D" (three fractions)?

**Technically yes, but not recommended.** The codebase includes a **simplex GP** option (`use_simplex_gp=True`) that takes inputs **(frac_MPC, frac_BMA, frac_MTAC)** with three length scales. However, this introduces **redundancy**:

- The ternary system has only **2 degrees of freedom** (sum = 1 constraint).
- Using 3D coordinates means the GP tries to learn in 3D space, but all data points lie on a **2D manifold** (the simplex).
- This can lead to **suboptimal learning** because the GP doesn't know about the constraint and may learn inappropriate correlations.

**Conclusion**: The original **2D GP (x, y)** is optimal. The 3D GP option exists for experimentation, but it is **not recommended** for production use. The fundamental issue (sparse x per y) remains regardless of the coordinate system used.
