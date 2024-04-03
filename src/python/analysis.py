from sympy import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker
import logging
import subprocess
import argparse 
import os.path
import pandas as pd 

log = logging.getLogger(__name__)

class PlotException(Exception):
    """Custom exception class with message for this module."""

    def __init__(self, value):
        self.value = value
        super().__init__(value)

    def __repr__(self):
        return repr(self.value)

def handle_command_line():
    """function to manage command line arguments """
    parser = argparse.ArgumentParser() 
    parser.add_argument('-f', '--folder',  default='./plots/',
                        help='folder to save plot', type=str)
    parser.add_argument('-b', '--b_value',  default=2.96848838e-2,
                        help='b_2 value', type=float)
    parser.add_argument('-i', '--runs_file', 
                        default='../../data/runs/GasLib-40-runs.csv', 
                        type=str)
    config = parser.parse_args()
    return config

def initialize():
    matplotlib.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size' : 12,
        'pgf.rcfonts': False,
        'text.latex.preamble': r"\usepackage{bm} \usepackage{amsmath}"
    })
    init_printing(use_unicode=True)
    
def crop(file): 
    subprocess.run(['pdfcrop', f'{file}.pdf'])
    subprocess.run(['rm', '-rf', f'{file}.pdf'])
    subprocess.run(['mv', f'{file}-crop.pdf', f'{file}.pdf'])
    
class Controller:
    """class that manages the functionality of the entire plotting script"""

    def __init__(self, config):
        self.config = config
        self.b_2 = config.b_value
        self.b_1 = 1.00300865
        self.c0 = None 
        self.c1 = None 
        self.c2 = None 
        self.al = 1.0
        self.au = 2.0
        self.xl = 3.0
        self.xu = 7.0
        self.a, self.x = symbols("a x", real=True)
        self.f1 = None 
        self.f2 = None 
        
    def least_square_rhs(self, v):
        return integrate(
            integrate(self.f2 * v, (self.a, self.al, self.au)
                      ), (self.x, self.xl, self.xu))
        
    def least_square_grammian(self, vi, vj):
        return integrate(
            integrate(
                vi * vj, (self.a, self.al, self.au)
                ), (self.x, self.xl, self.xu))
        
    def write_coefficients(self):
        f = open(f'{self.config.folder}/params.txt', 'w')
        print(f'b1 : {self.config.b_value}', file=f)
        print(f'c0 : {self.c0:.2f}', file = f)
        print(f'c1 : {self.c1:.2f}', file = f)
        print(f'c2 : {self.c2}', file = f)
        print(f'c3 : {self.c3}', file = f)
        f.close()
        
    def plot_alpha_comparision(self):
        alpha = np.linspace(1, 2, 100)
        fig, ax = plt.subplots()
        label = rf'${self.c2:.2f}\alpha^2 + {self.c3:.2f}\alpha^3$'
        ax.plot(alpha, self.c0 + 
                self.c1 * alpha + 
                self.c2 * np.power(alpha, 2) + 
                self.c3 * np.power(alpha, 3), label=label)
        ax.plot(alpha, np.power(alpha, 2), label=r'$\alpha^2$')
        ax.set_ylabel(r'$g(\alpha)$')
        ax.set_xlabel(r'$\alpha$')
        ax.legend(loc="best")
        plt.grid(alpha=0.3)
        fig.tight_layout()
        fig.set_size_inches(6, 4)
        plt.savefig(f'{self.config.folder}cnga_approx_vs_ideal_alpha.pdf', format='pdf')
        crop(f'{self.config.folder}cnga_approx_vs_ideal_alpha')
        
    def plot_errors(self):
        X = np.linspace(self.xl, self.xu, 500)
        Y = np.linspace(self.al, self.au, 500)
        xx, yy = np.meshgrid(X, Y)
        coeffs =[self.c0, self.c1, self.c2, self.c3]

        def pi_cnga(x):
            return (self.b_1)*np.power(x, 2)/2.0 + (self.b_2)*np.power(x,3)/3.0
        
        np_pi_cnga = np.vectorize(pi_cnga)
        
        def rel_err(x, alpha, coeffs):
            c0, c1, c2, c3 = coeffs
            return np.abs(
                np_pi_cnga(alpha*x) - 
                (c0 + c1*alpha + c2*(alpha**2) + c3*(alpha**3)) * np_pi_cnga(x)
                ) / np.abs(np_pi_cnga(alpha*x))

        zz1 = rel_err(xx, yy, coeffs)

        fig, ax = plt.subplots()
        plt.contourf(xx, yy, zz1.astype(float), 
                     locator=ticker.LogLocator(), cmap='inferno')
        plt.colorbar()
        plt.ylabel(r'$\alpha$', rotation='horizontal', ha='right')
        plt.xlabel(r'$p\; (\mathrm{MPa})$')
        plt.clim(vmin=1e-7, vmax=1)
        fig.tight_layout()
        fig.set_size_inches(5, 4)
        plt.savefig(f'{self.config.folder}opt_gamma.pdf')
        crop(f'{self.config.folder}opt_gamma')

        zz2 = rel_err(xx, yy, [0,0, 1.0, 0])

        fig, ax = plt.subplots()
        plt.contourf(xx, yy, zz2, locator=ticker.LogLocator(), cmap='inferno')
        plt.colorbar()
        plt.ylabel(r'$\alpha$', rotation='horizontal', ha='right')
        plt.xlabel(r'$p\; (\mathrm{MPa})$')
        plt.clim(vmin=1e-7, vmax=1)
        fig.tight_layout()
        fig.set_size_inches(5, 4)
        plt.savefig(f'{self.config.folder}alpha_sq_gamma.pdf')
        crop(f'{self.config.folder}alpha_sq_gamma')
        
    def plot_cdf(self): 
        if os.path.isfile(self.config.runs_file) == False: 
            log.info(f'{self.config.runs_file} does not exist')
            return 
        df = pd.read_csv(self.config.runs_file) 
        R_count, R_bins_count = np.histogram(df['R'], bins=50) 
        R_pdf = R_count # / sum(R_count) 
        R_cdf = np.cumsum(R_pdf) 
        E_count, E_bins_count = np.histogram(df['E'], bins=50) 
        E_pdf = E_count # / sum(E_count) 
        E_cdf = np.cumsum(E_pdf) 

        fig, ax = plt.subplots()
        ax.plot(R_bins_count[1:], R_cdf, '--', color='tab:red', linewidth=1.0, label=r"$||\bm R(\bm x_a)||_{\infty}$") 
        ax.plot(E_bins_count[1:], E_cdf, linewidth=1.0, label=r"$\dfrac{||\bm x_a -\bm x||_{\infty}}{||\bm x||_{\infty}}$")
        ax.legend(edgecolor='1.0', frameon=False)
        # ax.text(5*1e-4, 0.6, r'$\dfrac{||\bm x_a -\bm x||_{\infty}}{||\bm x||_{\infty}}$') 
        # ax.text(8*1e-3, 0.6, r'$||\bm R(\bm x_a)||_{\infty}$') 
        ax.set_xscale('log')
        ax.grid(alpha=0.1)
        ax.set_ylabel('Number of instances')
        ax.set_xlabel('Error')
        fig.tight_layout()
        fig.set_size_inches(5, 4)
        plt.savefig(f'{self.config.folder}cdf_GasLib40.pdf')
        crop(f'{self.config.folder}cdf_GasLib40')
        
    def run(self):
        self.f1 = (self.b_1/2) * (self.x**2) + (self.b_2/3) * (self.x**3)
        self.f2 = self.f1.subs(self.x, self.a * self.x)

        basis = [1 * self.f1, 
                 self.a * self.f1, self.a**2 * self.f1, 
                 self.a**3 * self.f1]
        index = list(range(0, len(basis)))
        
        b = Matrix(list(map(self.least_square_rhs, basis)))
        G = Matrix([ [self.least_square_grammian(basis[i], basis[j]) for j in index] for i in index ])


        solution = G.LUsolve(b)
        self.c0 = solution[0]
        self.c1 = solution[1]
        self.c2 = solution[2]
        self.c3 = solution[3]
        self.write_coefficients()
        log.info(f'c0: {self.c0:.2f}, c1: {self.c1:.2f}, c2: {self.c2:.2f}, c3: {self.c3:.2f}')
        
        self.plot_alpha_comparision()
        self.plot_errors()
        self.plot_cdf()

def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s--: %(message)s',
                        level=logging.INFO)
    
    initialize()
    try:
        config = handle_command_line()
        controller = Controller(config)
        controller.run()
    except PlotException as pe:
        log.error(pe)


if __name__ == "__main__":
    main()