import Mathlib

/-!
# Audit of the claimed 640/897 zeta-zero endgame

This file kernel-checks the self-contained algebra in Sections 4--6 of
`Zeta70_70percent_Proof.md` and records the defect in the manuscript's
Lemma 5.1.  It deliberately does not postulate the unresolved Gabor-channel
representation or the imported analytic number-theory statements.

There are no `sorry`, `admit`, or user-declared axioms.
-/

namespace Zeta70Audit

noncomputable section

/-- The polynomial used in manuscript equation (5.2). -/
def phi (sigma t : ℝ) : ℝ :=
  (1 - sigma) ^ 2 + 4 * (1 - sigma) * t - (((t - 1) ^ 2 - sigma) ^ 2)

/-- Manuscript identity (5.3). -/
lemma phi_factor (sigma t : ℝ) :
    phi sigma t =
      t * (8 * (1 - sigma) - 2 * (1 - sigma) * t - t * (t - 2) ^ 2) := by
  simp only [phi]
  ring

/-- Manuscript identity (5.4). -/
lemma phi_gap (sigma t : ℝ) :
    8 * (1 - sigma) - phi sigma t =
      (t - 2) ^ 2 * (t ^ 2 + 2 * (1 - sigma)) := by
  simp only [phi]
  ring

/-- Manuscript inequality (5.5). -/
lemma phi_nonpos_of_nonpos {sigma t : ℝ}
    (hsigma : sigma < 3 / 4) (ht : t ≤ 0) :
    phi sigma t ≤ 0 := by
  rw [phi_factor]
  have hc : 0 ≤ 1 - sigma := by nlinarith
  have h₁ : 0 ≤ 8 * (1 - sigma) := mul_nonneg (by norm_num) hc
  have h₂ : 0 ≤ -2 * (1 - sigma) * t := by
    have : 0 ≤ (2 * (1 - sigma)) * (-t) :=
      mul_nonneg (mul_nonneg (by norm_num) hc) (neg_nonneg.mpr ht)
    nlinarith
  have h₃ : 0 ≤ -t * (t - 2) ^ 2 :=
    mul_nonneg (neg_nonneg.mpr ht) (sq_nonneg (t - 2))
  have hbracket :
      0 ≤ 8 * (1 - sigma) - 2 * (1 - sigma) * t - t * (t - 2) ^ 2 := by
    nlinarith
  exact mul_nonpos_of_nonpos_of_nonneg ht hbracket

/-- Manuscript inequality (5.6). -/
lemma phi_le_eight {sigma t : ℝ} (hsigma : sigma < 3 / 4) :
    phi sigma t ≤ 8 * (1 - sigma) := by
  have hc : 0 ≤ 1 - sigma := by nlinarith
  have hprod : 0 ≤ (t - 2) ^ 2 * (t ^ 2 + 2 * (1 - sigma)) :=
    mul_nonneg (sq_nonneg (t - 2)) (by nlinarith [sq_nonneg t])
  nlinarith [phi_gap sigma t]

/-- Manuscript inequality (5.7); in fact no sign condition on `t` is needed. -/
lemma phi_le_linear (sigma t : ℝ) :
    phi sigma t ≤ (1 - sigma) ^ 2 + 4 * (1 - sigma) * t := by
  simp only [phi]
  nlinarith [sq_nonneg ((t - 1) ^ 2 - sigma)]

/-- The rational expression claimed in manuscript Lemma 5.1. -/
def claimedRatio (v k : ℝ) : ℝ :=
  (1 - v) ^ 2 / (1 - 2 * v + k)

/--
The scalar obstruction to Lemma 5.1 as stated: for `v = 4`, `k = 16`,
the claimed right-hand side is one.  The one-dimensional matrix `G = [-1]`
with `P = 0`, `Q = G`, `s = b = 0` satisfies (1.7)--(1.8), but has `s/N = 0`.
The manuscript's unconstrained optimizer is `sigma = 4`, outside `sigma < 3/4`.
-/
lemma lemma5_1_numeric_obstruction :
    claimedRatio (4 : ℝ) 16 = 1 ∧ ¬ claimedRatio (4 : ℝ) 16 ≤ 0 := by
  norm_num [claimedRatio]

/-- The formal optimizer used in the manuscript. -/
def sigmaStar (v k : ℝ) : ℝ :=
  (v - k) / (1 - v)

/-- At the target moments the optimizer is admissible. -/
lemma target_sigma :
    sigmaStar (1 / 3 : ℝ) (139 / 480) = 21 / 320 := by
  norm_num [sigmaStar]

lemma target_sigma_admissible :
    sigmaStar (1 / 3 : ℝ) (139 / 480) < 3 / 4 := by
  norm_num [sigmaStar]

/-- Exact value of the repaired zero-side ratio at the target moments. -/
lemma target_ratio :
    claimedRatio (1 / 3 : ℝ) (139 / 480) = 640 / 897 := by
  norm_num [claimedRatio]

/--
Correct replacement for the use of Lemma 5.1 in the final numerical endgame.
It fixes the admissible value `sigma = 21/320` instead of minimizing outside
its proved domain.
-/
theorem repaired_zero_side_target
    {r v k : ℝ}
    (hv : v = 1 / 3)
    (hk : k ≤ 139 / 480)
    (hmaster : ∀ sigma : ℝ, sigma < 3 / 4 →
      (1 - sigma) ^ 2 * (1 - r) ≤ sigma ^ 2 - 2 * v * sigma + k) :
    640 / 897 ≤ r := by
  have h := hmaster (21 / 320) (by norm_num)
  rw [hv] at h
  nlinarith

/-- The two direct alternating contractions sum to `1/10`. -/
lemma alternating_direct_mass :
    (1 / 20 : ℝ) + 1 / 20 = 1 / 10 := by
  norm_num

/-- The manuscript's proposed hard-wedge bookkeeping constant. -/
lemma alternating_mass_with_hard_wedge :
    (1 / 10 : ℝ) + 11 / 960 = 107 / 960 := by
  norm_num

/-- The proposed fourth centered moment constant, assuming the analytic inputs. -/
lemma fourth_moment_constant :
    4 * (1 / 60 : ℝ) + 2 * (1 / 10 + 11 / 960) = 139 / 480 := by
  norm_num

/-- The advertised final fraction is strictly above seventy percent. -/
lemma final_constant_gt_seventy :
    (7 / 10 : ℝ) < 640 / 897 := by
  norm_num

lemma final_margin :
    (640 / 897 : ℝ) - 7 / 10 = 121 / 8970 := by
  norm_num

/--
Conditional scalar endgame: once the manuscript's master zero-side inequality
and the analytic fourth-moment bound are supplied, the claimed constant follows.
-/
theorem conditional_640_897_endgame
    {r v k : ℝ}
    (hv : v = 1 / 3)
    (hk : k ≤ 4 * (1 / 60) + 2 * (1 / 10 + 11 / 960))
    (hmaster : ∀ sigma : ℝ, sigma < 3 / 4 →
      (1 - sigma) ^ 2 * (1 - r) ≤ sigma ^ 2 - 2 * v * sigma + k) :
    640 / 897 ≤ r := by
  have hk' : k ≤ 139 / 480 := by
    calc
      k ≤ 4 * (1 / 60) + 2 * (1 / 10 + 11 / 960) := hk
      _ = 139 / 480 := fourth_moment_constant
  exact repaired_zero_side_target (r := r) (v := v) (k := k) hv hk' hmaster

#print axioms phi_factor
#print axioms phi_gap
#print axioms phi_nonpos_of_nonpos
#print axioms phi_le_eight
#print axioms repaired_zero_side_target
#print axioms conditional_640_897_endgame

end

end Zeta70Audit
