# PDF_Fit
A code to fit the LN+PL or LN+2PL form to the density PDF of star forming regions.
To use this:

```python
pip install pdffit
````

Once you have installed, you can use the following sample script - 
```python
from fitter import *
import numpy as np


sample_data = np.load('./sample_data.npz')

#xdata and ydata are just two arrays. 
xdata = sample_data['arr_0']
ydata = sample_data['arr_1']
sink = float(sample_data['arr_2'])

params = Params(s_cut_off = sink)

p0 = [1.7, 1.6, 0.8, 7.1]
PLPLresult = PLPLFit(xdata, ydata, p0, params, use_K21=True)

p0 = [1.85, 1.57]
PLresult = PLFit(xdata, ydata, p0, params)

print (PLresult.sigma_err, PLresult.alpha_err, PLPLresult.sigma_err, PLPLresult.sb)
#To know more about how to access the result: help(Result)

```

In case you want to dig deeper or just use the function for plotting:
```python
from LNPLPL_functions import *
from LNPL_functions import *
```
or whichever way you wish to import the modules and the functions within them. 


To get help on any function:
```python
help(function_name)
```

If you're on the pypi page, please checkout the github version to get the sample data if you need. 