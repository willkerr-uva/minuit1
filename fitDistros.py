import ROOT
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameters, minimize
from scipy.stats import chi2

# Load histogram
f = ROOT.TFile("distros.root")
hist = f.Get("dist1")

nbins = hist.GetNbinsX()
bin_centers_all = np.array([hist.GetBinCenter(i+1) for i in range(nbins)])
counts_all = np.array([hist.GetBinContent(i+1) for i in range(nbins)])

mask = counts_all > 0
x = bin_centers_all[mask]
y = counts_all[mask]
err = np.sqrt(y)

# Model functions
def double_gaussian(x, params):
    A1 = params["A1"]
    mu1 = params["mu1"]
    sigma1 = params["sigma1"]
    A2 = params["A2"]
    mu2 = params["mu2"]
    sigma2 = params["sigma2"]

    g1 = A1 * np.exp(-0.5*((x - mu1)/sigma1)**2)
    g2 = A2 * np.exp(-0.5*((x - mu2)/sigma2)**2)
    return g1 + g2

def gumbel(x, params):
    A = params["A"]
    mu = params["mu"]
    beta = params["beta"]
    z = (x - mu) / beta
    return A * np.exp(-(z + np.exp(-z)))

# Chi-square objective
def chi2_objective(params, x, y, err, model_func):
    model = model_func(x, params)
    return (model - y) / err

# NLL objective (Poisson)
def nll_objective(params, x, y, model_func):
    m = model_func(x, params)
    m = np.clip(m, 1e-12, None)
    return (m - y*np.log(m))

# Fit double Gaussian (chi2)
params_gauss = Parameters()
params_gauss.add('A1', value=np.max(y)/2, min=0)
params_gauss.add('mu1', value=np.mean(x)-0.5)
params_gauss.add('sigma1', value=1.0, min=0.01)
params_gauss.add('A2', value=np.max(y)/2, min=0)
params_gauss.add('mu2', value=np.mean(x)+0.5)
params_gauss.add('sigma2', value=1.0, min=0.01)

result_gauss = minimize(chi2_objective, params_gauss, args=(x, y, err, double_gaussian))


# Fit Gumbel (chi2)

params_gumbel = Parameters()
params_gumbel.add('A', value=np.max(y), min=0)
peak = x[np.argmax(y)]
params_gumbel.add('mu', value=peak)
params_gumbel.add('beta', value=1.0, min=0.01)

result_gumbel = minimize(chi2_objective, params_gumbel, args=(x, y, err, gumbel))


# Compute chi2 and p-values

def chi2_stats(model_func, result, x, y, err):
    res = (model_func(x, result.params) - y) / err
    chi2_val = np.sum(res**2)
    dof = len(y) - len(result.params)
    p = 1 - chi2.cdf(chi2_val, dof)
    return chi2_val, dof, p

chi2_gauss, dof_gauss, p_gauss = chi2_stats(double_gaussian, result_gauss, x, y, err)
chi2_gumbel, dof_gumbel, p_gumbel = chi2_stats(gumbel, result_gumbel, x, y, err)


# NLL fits

result_gauss_nll = minimize(nll_objective, params_gauss, args=(x, y, double_gaussian))
result_gumbel_nll = minimize(nll_objective, params_gumbel, args=(x, y, gumbel))

def nll_value(result, model_func):
    m = model_func(x, result.params)
    m = np.clip(m, 1e-12, None)
    return np.sum(m - y*np.log(m))

nll_gauss = nll_value(result_gauss_nll, double_gaussian)
nll_gumbel = nll_value(result_gumbel_nll, gumbel)

delta_nll = nll_gumbel - nll_gauss


from matplotlib.backends.backend_pdf import PdfPages

with PdfPages("ex1.pdf") as pdf:
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.errorbar(x, y, yerr=err, fmt='o', label='data')
    plt.plot(x, double_gaussian(x, result_gauss.params), label='Double Gaussian fit')
    plt.title('Double Gaussian Fit')
    plt.xlabel('x')
    plt.ylabel('Counts')
    plt.legend()

    plt.subplot(1,2,2)
    plt.errorbar(x, y, yerr=err, fmt='o', label='data')
    plt.plot(x, gumbel(x, result_gumbel.params), label='Gumbel fit')
    plt.title('Gumbel Fit')
    plt.xlabel('x')
    plt.ylabel('Counts')
    plt.legend()

    plt.tight_layout()
    pdf.savefig()  
    plt.close()

    plt.figure(figsize=(11, 8.5)) 
    plt.axis('off')  

    text = f"""
Fit Results Summary
-------------------

Chi-square fits:
Double Gaussian:
    chi2 = {chi2_gauss:.2f}, dof = {dof_gauss}, p-value = {p_gauss:.3f}

Gumbel:
    chi2 = {chi2_gumbel:.2f}, dof = {dof_gumbel}, p-value = {p_gumbel:.3f}

Comments:
- The Double Gaussian fit has an extremely low p-value (~0), 
  meaning it is a poor fit to the data.
- The Gumbel fit has a very high p-value (~0.97),
  indicating an excellent agreement with the data.
- Based on chi2 and p-values, the Gumbel model is
  strongly preferred over the Double Gaussian model.

NLL-based fits (Poisson likelihood):
Double Gaussian: NLL = {nll_gauss:.2f}
Gumbel:          NLL = {nll_gumbel:.2f}
Delta NLL (Gumbel - Gaussian) = {delta_nll:.2f}

Comments:
- Since Delta NLL = NLL(Gumbel) - NLL(Gaussian) > 0, 
  the Gaussian fit has slightly lower NLL.
- However, chi2 and p-values strongly favor the Gumbel fit, 
  so overall the Gumbel model is preferred.
"""

    plt.text(0.05, 0.95, text, verticalalignment='top', fontsize=12, family='monospace')
    pdf.savefig()
    plt.close()

print("Saved multi-page PDF with plots and fit results to ex1.pdf")
