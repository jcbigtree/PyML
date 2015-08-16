"""Run benchmark tests on some or all available datasets
"""

# Author: Jinchao Liu <liujinchao2000@gmail.com>
# License: BSD 

from pylab import *
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import time
import subprocess
import os
import tempfile
import shutil
import codecs
import copy

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold
from sklearn.preprocessing import normalize

from mlnp.datasets import load_dataset
from mlnp.cross_validation import train_test_split
from mlnp.preprocessing import standardize
from mlnp.preprocessing import normalize
from mlnp.report.latex_template import *

__all__=[
    "run_benchmark", "plot_barchart"
]

#################################################################################################
# GLOBAL VARIABLES
g_benchmark_datasets = ['Abalone2c', 
                        'Australian', 
                        'German', 
                        'Heart', 
                        'Ionosphere', 
                        'Pima', 
                        'Banana']

g_result_top_dir = [] #Value is assigned in the body of run_benchmark

#################################################################################################
def run_benchmark(learners_obj,
                  learners_name,  # For generating reports
                  n_runs = 30,
                  n_folds = 5,
                  noise_level = 0.0,
                  preprocessing = 'none',  
                  report_name="benchmark.pdf",
                  verbose=False):    
    """Run benchmark tests to evaluate a method and generate a latex-compatible report.
    
    Parameters:
    -----------
    learners_obj: a set of object. list or tuple 
        objects of methods to be tested
    
    learners_name: a set of string. list of tuple
    
    n_runs: int default 30
        Runs of cross-validation 
    
    n_folds: int default 5
        Number of folds. Must be at least 2. Typically 5 or 10
    
    noise_level: number default 0.0
        Percentage of samples labels of which are flipped    
    
    preprocessing: str. default 'none'
        Preprocessing method. Must be 'none' or 'normalize' or 'standardize' 
        
    report_name: string default "benchmark.txt"    
        Name of a report that will be generated.
        
    verbose: bool default False
        If true, display progress 
    """
    
    # Check parameters 
    
    
    # Run benchmark tests
    # ----------------------------------------------------------------------------#    
    #             Result format                                                   #
    #                                                                             #
    #    ROW 0 : ds_name, run,  cv_fold, method_index, train_error, test_error    # 
    #    ROW 1 : ds_name, run,  cv_fold, method_index, train_error, test_error    #
    #                                                                             #
    #    ROW k : ds_name, run,  cv_fold, method_index, train_error, test_error    #
    #                                                                             #
    # ----------------------------------------------------------------------------#
    
    benchmark_results = []
    ds_count = -1
    for ds in g_benchmark_datasets:
        ds_count += 1
        X, y, n_samples, n_features = _load_data(ds, noise_level)
        for irun in range(n_runs):                
            # Cross-validation
            kf = KFold(n_samples, n_folds=n_folds, shuffle=True)
            fold_count = -1                    
            
            for train_idx, test_idx in kf:
                fold_count += 1
                X_train = X[train_idx,:]
                y_train = y[train_idx]
                X_test = X[test_idx,:]    
                y_test = y[test_idx]
                
                if preprocessing == 'standardize':
                    X_train = standardize(X_train)
                    X_test = standardize(X_test)                

                if preprocessing == 'normalize':
                    X_train = normalize(X_train)
                    X_test = normalize(X_test)
                    
                
                # Call learners 
                method_count = -1
                for learner in learners_obj:
                    method_count += 1                    
                    lcopy = copy.deepcopy(learner)                                        
                    lcopy.fit(X_train,y_train)                    
                    train_err = 1.0 - lcopy.score(X_train, y_train)              
                    test_err = 1.0 - lcopy.score(X_test, y_test)                   
                    
                    result = [ds_count, irun, fold_count, method_count, train_err, test_err]
                    benchmark_results.append(result)
                                             
                    if verbose: 
                        print 'dataset: {0}  run: {1}  cv_fold: {2}  ' \
                              'method: {3}  train_error: {4}  test_error: {5}' \
                                .format(ds, irun, fold_count, \
                                        method_count, train_err, test_err)
                        #print result
                        
    # Save results
    print 'Saving results ... '
    global g_result_top_dir
    g_result_top_dir = os.path.join(os.getcwd(), "benchmark_results")
    if not os.path.exists(g_result_top_dir):    
        os.mkdir(g_result_top_dir)
       
    if not (".pdf" in report_name.lower()):
        report_name += '.pdf'    
    dir_name = report_name.replace('.pdf','') 
    if not os.path.exists(os.path.join(g_result_top_dir, dir_name)):    
        os.mkdir(os.path.join(g_result_top_dir, dir_name))       
    

    datetime = time.strftime("%H%M%S%d_%m_%Y")
    result_file_name = os.path.join(os.path.join(g_result_top_dir, dir_name), 
                             "benchmark_results_" + datetime)      
    np.save(result_file_name, benchmark_results)
    
    # Save meta information: names of datasets and methods for generating a report.
    np.save(result_file_name+'_methods_name', learners_name)
    np.save(result_file_name+'_datasets_name', g_benchmark_datasets)                
        
    # Generate report
    print 'Generating reports ... '
    _generate_report(report_name=report_name, 
                     result_file=result_file_name,
                     methods_name= learners_name,
                     n_runs=n_runs, 
                     n_folds=n_folds) 
    
    
    
    
#################################################################################################
def _load_data(ds_name, noise_level=0.0): 
    """Internal function for loading datasets"""
    X, y = load_dataset(ds_name, noise_level=noise_level)
    X = np.array(X)                
    y = np.array(y)
    
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_positive_samples = 0
    n_negative_samples = 0
    
    # Preprocessing labels
    y_labels = []
    for l in y:
        if not l in y_labels:
            y_labels.append(l)
            
    for yi in y:
        if yi == -1: n_negative_samples += 1
        if yi ==  1: n_positive_samples += 1
    
    
    # Print the dataset infomation
    print '-------------------------------------------------------'
    print 'Dataset:                     ', ds_name
    print 'Number of Instances:         ', X.shape[0]
    print 'Number of Attributes:        ', X.shape[1]
    print 'Number of Classes:           ', len(y_labels)
    print 'Number of Positive Instance: ', n_positive_samples
    print 'Number of Negative Instance: ', n_negative_samples
    print '-------------------------------------------------------'
    
    return X, y, n_samples, n_features
      


#################################################################################################
# PLOT RESULTS
def plot_barchart(result_file):    
    """Plot results as bar chart.
    
    Parameters:
    -----------
    result_file: npy file 
        where results are stored.    
    """    
    
    #ROW 0 : ds_name, run,  cv_fold, method_index, train_error, test_error  
    if '.npy' in result_file:
        result_file = result_file.replace('.npy','')        
        
    results = np.load(result_file+'.npy')
    methods_name = np.load(result_file+'_methods_name'+'.npy')
    datasets_name = np.load(result_file+'_datasets_name'+'.npy')
    n_methods = len(methods_name)
    n_datasets = len(datasets_name)
    
    stats = []    
    for i in range(n_methods):
        method_stats = []        
        for j in range(n_datasets):
            ds_results = results[np.where(results[:,0]==j)[0],:]            
            result = ds_results[np.where(ds_results[:,3]==i)[0],:]            
            training_error = result[:,4]
            test_error = result[:,5]
            method_stats.append([np.mean(test_error), np.std(test_error)])
        stats.append(method_stats)             
        
    stats_arr = np.array(stats) # of shape [n_methods, n_datasets, 2]
    max_err = np.max(stats_arr[:,:,0])
    
    # Plot bars to compare methods 
    index = np.arange(n_datasets)            # the x locations for the groups
    bar_width = np.min(0.75/n_methods, 0.2)  # the width of the bars

    #matplotlib.rc('font',**{'family':'sans-serif','sans-serif':'Helvetica','size':16})
    #matplotlib.rc('font',**dict(family='serif',serif='Palatino',size='16'))
    matplotlib.rc('font',**dict(family='serif',serif='Palatino'))
    matplotlib.rc('text', usetex=True)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    opacity = 0.65
    
    for i in range(0, n_methods):   
        data = np.array(stats[i])
        ax.bar(index+i*bar_width, data[:,0], bar_width, alpha=opacity, 
               label=methods_name[i], color=cm.jet(1.0*i/n_methods))
    
    plt.ylim(0, max_err*(1+n_methods*0.11))
    plt.ylabel('Misclassification Error')
    plt.title('Benchmark test on UCI Datasets')
    plt.xticks(index + bar_width*1.75, g_benchmark_datasets)
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    #plt.show()
    plt.savefig(result_file+'_barchart.pdf')   



#################################################################################################
# GENERATE REPORT 
def _generate_report(report_name, 
                     result_file, 
                     methods_name,
                     n_runs,
                     n_folds):
    """Generate a latex-compatible report."""
    if '.npy' in result_file:
        result_file = result_file.replace('.npy','')        
        
    results = np.load(result_file+'.npy')    
    results = np.array(results)
    n_methods = len(methods_name)
    
    stats = []    
    for i in range(n_methods):
        method_stats = []        
        for ds_index in range(len(g_benchmark_datasets)):
            ds_results = results[np.where(results[:,0]==ds_index)[0],:]            
            result = ds_results[np.where(ds_results[:,3]==i)[0],:]            
            training_error = result[:,4]
            test_error = result[:,5]
            #print "{:0.4f}".format(np.mean(test_errors)), "({:0.4f})".format(np.std(test_errors))
            method_stats.append([np.mean(test_error), np.std(test_error)])
        stats.append(method_stats)          

    table_data = []
    table_data.append(['method'] + g_benchmark_datasets)

    for i in range(0, n_methods):
        row_data = [methods_name[i]]
        for ds_index in range(0, len(g_benchmark_datasets)):            
            err = stats[i][ds_index]            
            row_data.append("{:0.4f}".format(err[0]) + '(' + "{:0.4f}".format(err[1]) + ')')
            
        table_data.append(row_data)                
            
    my_latex_table = fill_latex_table(table_data=table_data,
                                      caption="Benchmark test on UCI Datasets")              
    
    
    # Plot bar chart 
    plot_barchart(result_file)
    if '.npy' in result_file:
        result_file = result_file.replace('.npy','')
        
    result_file += '_barchart.pdf'   
    result_file = result_file.replace('\\','/')    
    my_latex_figure = create_figure(figure=result_file, 
                                    width=0.75, 
                                    caption = 'Benchmark test on UCI Datasets', 
                                    label='Fig:BenchmarkTest')
  
    tex = doc_pre_template + my_latex_table + my_latex_figure + doc_post_template    
    _generate_pdf(report_name, tex)
            


#################################################################################################
# GENERATE PDF
def _generate_pdf(pdf_name,tex):
    """Genertates the pdf from string"""
    
    if not (".pdf" in pdf_name.lower()):
        pdf_name += '.pdf'    
    tex_name = pdf_name.replace('.pdf','.tex')
    dir_name = pdf_name.replace('.pdf','')
    
    current = os.getcwd()
    temp = tempfile.mkdtemp()
    os.chdir(temp)
    
    f = open('tem_report.tex','w')
    f.write(tex)
    f.close()
    
    #with codecs.open("cover.tex",'w',encoding='utf8') as f:
    #    f.write(tex)
    
    proc=subprocess.Popen(['pdflatex','tem_report.tex'])
    subprocess.Popen(['pdflatex',tex])
    proc.communicate()    
            
    if not os.path.exists(os.path.join(g_result_top_dir, dir_name)):    
        os.mkdir(os.path.join(g_result_top_dir, dir_name))
          
    os.rename('tem_report.pdf',pdf_name)
    shutil.copy(pdf_name, os.path.join(g_result_top_dir, dir_name))
    
    os.rename('tem_report.tex',tex_name)
    shutil.copy(tex_name, os.path.join(g_result_top_dir, dir_name))
        
    os.chdir(current)
    shutil.rmtree(temp)
    

