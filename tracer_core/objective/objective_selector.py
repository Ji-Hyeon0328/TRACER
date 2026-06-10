class ObjectiveSelector(object):
    """
    M8: Rule-based Objective Selector shell.

    Input:
        posture severity
        rho_v_mean
        sigma_v

    Output:
        mode
        beta = [beta_v, beta_s, beta_e]

    Later, this module can be replaced by a learned objective selector.
    """

    def __init__(self,
                 mismatch_soft=0.50,
                 mismatch_hard=0.80,
                 sigma_soft=0.18,
                 sigma_hard=0.35,
                 posture_soft=False):
        self.mismatch_soft = mismatch_soft
        self.mismatch_hard = mismatch_hard
        self.sigma_soft = sigma_soft
        self.sigma_hard = sigma_hard
        self.posture_soft = posture_soft

    def select(self, posture_conservative, rho_v_mean, sigma_v):
        if posture_conservative:
            return {
                "mode": "conservative_posture",
                "beta_v": 0.20,
                "beta_s": 0.70,
                "beta_e": 0.10,
            }

        if rho_v_mean > self.mismatch_hard or sigma_v > self.sigma_hard:
            return {
                "mode": "conservative_mismatch",
                "beta_v": 0.25,
                "beta_s": 0.45,
                "beta_e": 0.30,
            }

        if rho_v_mean > self.mismatch_soft or sigma_v > self.sigma_soft:
            return {
                "mode": "cautious_mismatch",
                "beta_v": 0.45,
                "beta_s": 0.35,
                "beta_e": 0.20,
            }

        return {
            "mode": "nominal",
            "beta_v": 0.70,
            "beta_s": 0.20,
            "beta_e": 0.10,
        }
