import numpy as np
from scipy.integrate import quad
import PyMieScatt as ps
import matplotlib.pyplot as plt


class SingleScatteringSim:
    """
    基于Mie理论和Beer-Lambert定律的单次散射数值模拟模型。
    该模型用于计算从悬浮液侧向散射（90度）的光的退偏比（DPR）。
    """

    def __init__(self, particle_diameter_nm, wavelength_nm, n_particle, n_medium,
                 particle_concentration_per_mm3, sample_height_mm, mu_a_per_mm=0.0):
        """
        初始化模拟参数。

        参数:
        - particle_diameter_nm (float): 球形粒子的直径 (单位: 纳米 nm)。
        - wavelength_nm (float): 入射光的波长 (单位: 纳米 nm)。
        - n_particle (complex): 粒子的复折射率 (例如: 1.59 + 0.0j)。
        - n_medium (float): 周围介质的折射率 (例如: 水的折射率约 1.33)。
        - particle_concentration_per_mm3 (float): 粒子数浓度 (单位: 个/立方毫米 mm^3)。
        - sample_height_mm (float): 样品的高度/总光程 (单位: 毫米 mm)。
        - mu_a_per_mm (float): 介质的吸收系数 (单位: 1/mm)。
        """
        self.d = particle_diameter_nm
        self.wav = wavelength_nm
        self.n_p = n_particle
        self.n_m = n_medium
        self.concentration = particle_concentration_per_mm3
        self.H = sample_height_mm
        self.mu_a = mu_a_per_mm

        # 内部计算变量
        self.mu_s = None  # 散射系数 (1/mm)
        self.mu_t = None  # 总消光系数 (1/mm)
        self.S1_90 = None  # 90度角的S1振幅
        self.S2_90 = None  # 90度角的S2振幅

    def _calculate_mie_properties(self):
        """
        对应原理第一部分：Mie散射理论
        计算核心的Mie散射参数。
        """
        # 1. 计算单个粒子的散射截面 Csca
        # PyMieScatt.MieQ 返回散射效率Qsca, 需乘以粒子几何截面积得到散射截面Csca
        radius_nm = self.d / 2.0
        geo_cross_section_nm2 = np.pi * radius_nm ** 2

        # Qsca是无量纲的效率因子
        q_sca, _, _, _, _, _, _ = ps.MieQ(self.n_p, self.wav, self.d, nMedium=self.n_m, asDict=False)

        # Csca 单位是 nm^2
        csca_nm2 = q_sca * geo_cross_section_nm2
        # 将Csca单位从 nm^2 转换为 mm^2，以匹配浓度的单位
        csca_mm2 = csca_nm2 * (1e-6) ** 2

        # 2. 计算宏观散射系数 μ_s (mu_s)
        # μ_s = 浓度 * Csca，单位是 1/mm
        self.mu_s = self.concentration * csca_mm2

        # 3. 计算总消光系数 μ_t (mu_t)
        # 对应原理第二部分：μ_t = μ_a + μ_s
        self.mu_t = self.mu_a + self.mu_s

        # 4. 计算90度方向的散射振幅函数 S1 和 S2
        # ps.ScatterinFunction需要角度的余弦值，90度角 cos(90) = 0
        theta_rad = np.pi / 2
        mu = np.cos(theta_rad)
        S1, S2, Qext, Qsca = ps.ScatteringFunction(self.n_p, self.wav, self.d, self.n_m, mu)

        self.S1_90 = S1
        self.S2_90 = S2

        print("--- Mie散射参数计算完成 ---")
        print(f"散射系数 μ_s: {self.mu_s:.4f} /mm")
        print(f"总消光系数 μ_t: {self.mu_t:.4f} /mm")
        print(f"90度散射振幅 |S₁|²: {np.abs(self.S1_90[0]) ** 2:.4e}")  # 添加 [0]
        print(f"90度散射振幅 |S₂|²: {np.abs(self.S2_90[0]) ** 2:.4e}")  # 添加 [0]

    def _integrand(self, z, s_squared):
        """
        对应原理中积分核：η(z,λ) * I_in(λ,z) * p(θ=90°,λ)

        - η(z,λ): 假设粒子浓度均匀，为常数，可以在最终比值中约去。
        - I_in(λ,z): 由Beer-Lambert定律给出，I_in = I₀ * exp(-μ_t * z)。I₀也可约去。
        - p(θ=90°,λ): 角向散射概率，正比于 |S|²。

        参数:
        - z (float): 深度 (mm)。
        - s_squared (float): |S₁|² 或 |S₂|²。
        """
        # I_in(z) / I₀
        attenuation = np.exp(-self.mu_t * z)
        # 整个被积函数（已约去常数项）
        return attenuation * s_squared

    def run_simulation(self):
        """
        执行完整的模拟流程：计算Mie参数 -> 执行积分 -> 计算DPR。
        """
        # 步骤 1: 计算所有需要的Mie参数
        self._calculate_mie_properties()

        # 步骤 2: 执行数值积分
        # 对应原理第三部分：I_scat(λ) = ∫[0 to H] ... dz
        # 我们分别对垂直和水平分量进行积分

        # 计算垂直分量总散射强度 I_scat_perp
        # 计算平行分量总散射强度 I_scat_para
        i_scat_perp, err_perp = quad(self._integrand, 0, self.H, args=(np.abs(self.S1_90[0]) ** 2,))
        i_scat_para, err_para = quad(self._integrand, 0, self.H, args=(np.abs(self.S2_90[0]) ** 2,))

        print("\n--- 积分计算完成 ---")
        print(f"总垂直散射强度 (积分值): {i_scat_perp:.4e}")
        print(f"总平行散射强度 (积分值): {i_scat_para:.4e}")

        # 步骤 3: 计算最终的DPR
        # 对应原理最后一句：最终的 DPR 由平行与垂直方向的散射光强之比给出
        if i_scat_perp == 0:
            final_dpr = float('inf')  # 避免除以零
        else:
            final_dpr =  i_scat_perp/i_scat_para

        return final_dpr


# --- 模型使用示例 ---
if __name__ == "__main__":
    # 定义模拟参数 (例如：水中的聚苯乙烯微球)
    params = {
        "particle_diameter_nm": 500,  # 粒子直径 500nm
        "wavelength_nm": 633,  # He-Ne激光波长 633nm
        "n_particle": 1.59,  # 聚苯乙烯在633nm的折射率 (假设无吸收)
        "n_medium": 1.33,  # 水的折射率
        "particle_concentration_per_mm3": 1e7,  # 10^7 个粒子/mm^3
        "sample_height_mm": 10,  # 样品池高度 10mm (1cm)

        "mu_a_per_mm": 0.001  # 假设一个很小的背景吸收
    }

    # 创建模拟器实例
    simulator = SingleScatteringSim(**params)

    # 运行模拟并获取结果
    dpr_result = simulator.run_simulation()

    print("\n--- 模拟结果 ---")
    print(f"模型参数: {params}")
    print(f"计算得到的退偏比 (DPR): {dpr_result:.6f}")

    # (可选) 绘制被积函数随深度的变化
    z_vals = np.linspace(0, params["sample_height_mm"], 200)
    # 获取S1和S2在90度角的标量值
    # np.abs(simulator.S1_90) ** 2 仍然可能返回一个 (1,) 形状的数组
    # 我们需要确保它是标量，所以再次使用 [0] 索引或 .item()
    s1_squared_90 = np.abs(simulator.S1_90[0]) ** 2
    s2_squared_90 = np.abs(simulator.S2_90[0]) ** 2

    integrand_perp = simulator._integrand(z_vals, s1_squared_90)
    integrand_para = simulator._integrand(z_vals, s2_squared_90)

    plt.figure(figsize=(10, 6))
    plt.plot(z_vals, integrand_perp, label=r'Integrand for $I_{\perp}$ (proportional to $|S_1|^2$)')
    plt.plot(z_vals, integrand_para, label=r'Integrand for $I_{||}$ (proportional to $|S_2|^2$)')
    plt.xlabel("Depth z (mm)")
    plt.ylabel("Integrand Value (arb. units)")
    plt.title("Contribution to Scattered Light vs. Depth")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()