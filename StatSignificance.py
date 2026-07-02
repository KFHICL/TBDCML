# %%
# update
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, ttest_rel, mannwhitneyu, ttest_ind, normaltest

class CrossValidationStatTest:
    def __init__(self, model1_nrmse, model2_nrmse):
        """
        Initialize with NRMSE values for two models
        
        Args:
            model1_nrmse: List/array of NRMSE values for model 1
            model2_nrmse: List/array of NRMSE values for model 2
        """
        self.model1_nrmse = np.array(model1_nrmse)
        self.model2_nrmse = np.array(model2_nrmse)
        
    def paired_ttest(self):
        """
        Paired t-test (assumes same CV folds for both models)
        H0: mean difference = 0
        H1: model1 has significantly different NRMSE than model2
        """
        statistic, p_value = ttest_rel(self.model1_nrmse, self.model2_nrmse)
        return statistic, p_value
    
    def wilcoxon_test(self):
        """
        Wilcoxon signed-rank test (non-parametric paired test)
        H0: median difference = 0
        H1: model1 has significantly different NRMSE than model2
        """
        statistic, p_value = wilcoxon(self.model1_nrmse, self.model2_nrmse)
        return statistic, p_value
    
    def unpaired_ttest(self):
        """
        Independent t-test (if models trained on different folds)
        H0: mean NRMSE equal
        H1: model1 has significantly different NRMSE than model2
        """
        statistic, p_value = ttest_ind(self.model1_nrmse, self.model2_nrmse, equal_var=False, alternative='less')
        return statistic, p_value
    
    def mann_whitney_test(self):
        """
        Mann-Whitney U test (non-parametric independent test)
        H0: distributions equal
        H1: model1 has significantly different NRMSE than model2
        """
        statistic, p_value = mannwhitneyu(self.model1_nrmse, self.model2_nrmse, alternative='two-sided')
        return statistic, p_value
    
    def run_all_tests(self, alpha=0.05):
        """
        Run all statistical tests and return results
        """
        results = {}
        
        # Paired tests (recommended for CV)
        t_stat, t_p = self.paired_ttest()
        w_stat, w_p = self.wilcoxon_test()
        
        # Unpaired tests
        ind_t_stat, ind_t_p = self.unpaired_ttest()
        mw_stat, mw_p = self.mann_whitney_test()
        
        results = {
            'paired_ttest': {'statistic': t_stat, 'p_value': t_p, 'significant': t_p < alpha},
            'wilcoxon': {'statistic': w_stat, 'p_value': w_p, 'significant': w_p < alpha},
            'unpaired_ttest': {'statistic': ind_t_stat, 'p_value': ind_t_p, 'significant': ind_t_p < alpha},
            'mann_whitney': {'statistic': mw_stat, 'p_value': mw_p, 'significant': mw_p < alpha}
        }
        
        return results
    
    def summary_stats(self):
        """
        Calculate summary statistics
        """
        return {
            'model1': {
                'mean': np.mean(self.model1_nrmse),
                'std': np.std(self.model1_nrmse),
                'median': np.median(self.model1_nrmse)
            },
            'model2': {
                'mean': np.mean(self.model2_nrmse),
                'std': np.std(self.model2_nrmse),
                'median': np.median(self.model2_nrmse)
            }
        }

# Example usage:

LFC18_Baseline_Val_NRMSE = [
    0.04296919, 0.043893377, 0.043553576, 0.04482441, 0.043788209,
    0.044637977, 0.044537536, 0.043813265, 0.040300641, 0.043234208,
    0.040539889, 0.044932022, 0.046844147, 0.044848059, 0.045062391,
    0.045830864, 0.041573016, 0.044721037, 0.04183666, 0.046009296,
    0.043792335, 0.044720463, 0.045363114, 0.041881603, 0.041711878,
    0.044044871, 0.045617238, 0.044646051, 0.042451257, 0.042144892
]

LFC18_Opti_Val_NRMSE = [
    0.012383067, 0.012858508, 0.012012376, 0.012540281, 0.01305019,
    0.01186376, 0.012727997, 0.013128691, 0.014521738, 0.012938172,
    0.011790244, 0.01242047, 0.012093111, 0.012325373, 0.012877263,
    0.012657205, 0.012529486, 0.013122384, 0.01295331, 0.013477851,
    0.011864116, 0.012914952, 0.012017672, 0.012783557, 0.012562299,
    0.011943051, 0.013082643, 0.013102904, 0.013849316, 0.013714156
]

LFC18_Baseline_Train_RMSE = [
    0.11393029, 0.103018376, 0.1106546, 0.109859547, 0.10999859,
    0.117748317, 0.10165425, 0.102892905, 0.10435521, 0.087737541,
    0.081104637, 0.112945961, 0.123475158, 0.115993188, 0.099534488,
    0.129994903, 0.094583615, 0.085559847, 0.093366948, 0.137533641,
    0.092446308, 0.107057265, 0.130901259, 0.103584375, 0.110551783,
    0.101915457, 0.109970828, 0.09473951, 0.092953913, 0.089497558
]

LFC18_Baseline_Val_RMSE = [
    0.161993847, 0.165478032, 0.164196981, 0.168988024, 0.165081549,
    0.168285172, 0.167906511, 0.165176007, 0.151933417, 0.162992965,
    0.152835383, 0.169393722, 0.176602432, 0.169077182, 0.169885215,
    0.172782356, 0.156730271, 0.168598309, 0.157724209, 0.173455047,
    0.165097102, 0.168596144, 0.17101894, 0.157893643, 0.157253778,
    0.166049163, 0.171976986, 0.168315612, 0.160041239, 0.158886241
]


LFC18_Opti_Val_RMSE = [
    0.144643077, 0.134007644, 0.125165174, 0.129085077, 0.129983934,
    0.135332464, 0.13952321, 0.130698596, 0.132385929, 0.130912357,
    0.133530862, 0.136055044, 0.127988576, 0.134024267, 0.133384176,
    0.132548301, 0.135980822, 0.141120917, 0.12766626, 0.130658729,
    0.128856584, 0.130538433, 0.133228562, 0.131766018, 0.132416309,
    0.131181108, 0.133142853, 0.140157558, 0.12242264, 0.134913094
]

LFC18_MSE_Val_RMSE = 0.1300308631
from scipy.stats import ttest_1samp
res = ttest_1samp(LFC18_Opti_Val_RMSE,LFC18_MSE_Val_RMSE, alternative='greater')

LFC18_trainVal = ttest_rel(LFC18_Baseline_Train_RMSE, LFC18_Baseline_Val_RMSE, axis=0,  alternative='less')



MC24_Baseline_Train_RMSE = [
    0.083165675, 0.085977927, 0.084610537, 0.084089826, 0.078916799,
    0.084859039, 0.082337147, 0.082321143, 0.08674728, 0.081501813,
    0.081470377, 0.085919086, 0.082462655, 0.086706557, 0.081907892,
    0.087551573, 0.094277858, 0.078589036, 0.085944715, 0.084117731,
    0.079082396, 0.0855944, 0.08570047, 0.081230682, 0.086581637,
    0.089129444, 0.080918019, 0.088598831, 0.077864143, 0.089383939
]

MC24_Baseline_Val_RMSE = [
    0.079132581, 0.091644927, 0.09245559, 0.079024053, 0.099469091,
    0.093753835, 0.094126799, 0.074175819, 0.08648284, 0.076463045,
    0.081142651, 0.08255543, 0.100478161, 0.09283044, 0.077234549,
    0.09475309, 0.086513356, 0.078899054, 0.081698372, 0.072567872,
    0.101295671, 0.102700851, 0.091376162, 0.096390054, 0.082120676,
    0.081667571, 0.109629646, 0.070029737, 0.090847801, 0.07560141
]


MC24_trainVal = ttest_rel(MC24_Baseline_Train_RMSE, MC24_Baseline_Val_RMSE, axis=0,  alternative='two-sided')


MC24_1000_Baseline_Train_RMSE = [
    0.085768322, 0.085292719, 0.08610604, 0.085180298, 0.087014352,
    0.084804635, 0.085704393, 0.085443017, 0.084193762, 0.083108022,
    0.085645285, 0.08567427, 0.08635486, 0.083334309, 0.08461401,
    0.086637604, 0.085947155, 0.084139033, 0.086339738, 0.08487697,
    0.085786799, 0.086343829, 0.086825907, 0.085767111, 0.085511248,
    0.084577856, 0.086664527, 0.084927081, 0.085427908, 0.084909382
]

MC24_1000_Baseline_Val_RMSE = [
    0.085337866, 0.088553273, 0.083401878, 0.086940614, 0.083299193,
    0.084158996, 0.08518527, 0.088397588, 0.090685673, 0.092126725,
    0.084503254, 0.088032375, 0.082909175, 0.088029125, 0.080974954,
    0.089127876, 0.087256656, 0.089822175, 0.089653644, 0.092322725,
    0.085953117, 0.087342996, 0.084036794, 0.087344014, 0.084246805,
    0.087061941, 0.087740131, 0.091966007, 0.089157192, 0.091351209
]

MC24_1000_trainVal = ttest_rel(MC24_1000_Baseline_Train_RMSE, MC24_1000_Baseline_Val_RMSE, axis=0,  alternative='less')


MC24_1000_Baseline_Val_NRMSE = [
    0.101592698, 0.105420563, 0.09928795, 0.103500731, 0.099165706,
    0.100189281, 0.101411036, 0.105235223, 0.107959135, 0.109674673,
    0.100599112, 0.104800447, 0.098701399, 0.104796577, 0.096398754,
    0.106104614, 0.103876972, 0.106931161, 0.106730528, 0.109908006,
    0.10232514, 0.103979757, 0.100043803, 0.103980969, 0.100293815,
    0.103645168, 0.104452537, 0.109483341, 0.106139514, 0.108751439
]

MC24_1000_Opti_Val_NRMSE = [
    0.055576386, 0.057710208, 0.05391269, 0.056281974, 0.058570495,
    0.053245687, 0.057124464, 0.058922814, 0.065174941, 0.058067749,
    0.052915737, 0.055744252, 0.054275035, 0.055317447, 0.057794381,
    0.056806742, 0.056233525, 0.058894509, 0.058135688, 0.060489878,
    0.053247283, 0.057963536, 0.053936456, 0.057373821, 0.056380797,
    0.05360155, 0.058716148, 0.058807079, 0.062157048, 0.061550439
]


MC24_1000_Opti_Train_RMSE = [
0.057361246,
0.056773496,
0.056813463,
]
MC24_1000_Opti_Test_RMSE = [
                            0.05748351,
0.053759153,
0.055248885,
]

MC24_1000_trainTest = ttest_rel(MC24_1000_Opti_Train_RMSE, MC24_1000_Opti_Test_RMSE, axis=0,  alternative='greater')



normal_1 = normaltest(LFC18_Opti_Val_NRMSE)
normal_2 = normaltest(MC24_1000_Opti_Val_NRMSE)

test = CrossValidationStatTest(LFC18_Opti_Val_NRMSE, MC24_1000_Opti_Val_NRMSE)
results = test.run_all_tests()
stats = test.summary_stats()


MC24x_Opti_Val_RMSE = [0.055630605,
0.057629028,
0.058640221,
]

MC24x_Opti_Train_RMSE = [0.057361246,
0.056773496,
0.056813463,
]

MC24x_Opti_Test_RMSE = [0.05748351,
0.053759153,
0.055248885,
]

MC24x_trainVal = ttest_rel(MC24x_Opti_Train_RMSE, MC24x_Opti_Val_RMSE, axis=0,  alternative='less')
MC24x_trainTest = ttest_rel(MC24x_Opti_Test_RMSE, MC24x_Opti_Train_RMSE, axis=0,  alternative='less')

# %%
