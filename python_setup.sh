# Open a terminal and check if you have « nibabel » and « dipy » python modules :
ipython
import nibabel
import dipy
import nilearn

# If not, install each missing module. Ex:
# Close ipython (Ctrl+d and y)
# In a bash terminal execute the following code :
pip install --user dipy
pip install --user nibabel

# set Anaconda path
export PATH=/cal/softs/anaconda/anaconda2/bin:$PATH

# To launch ipython notebook:
# Close ipython (Ctrl+d and y)
# Copy the practical directory ('BME_DWMRI') at the root of your home directory
# Go in this BME_DWMRI directory by executing :
cd ~/BME_DWMRI

# And launch the notebook using jupyter :
jupyter notebook bme_dwi_practical.ipynb
