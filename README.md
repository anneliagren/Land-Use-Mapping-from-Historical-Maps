# Land-Use-Mapping-from-Historical-Maps
The methods description is now in review. After publication a link to the reserch article will be published HERE: 

![GraphicalAbstract](https://github.com/user-attachments/assets/a3b9da90-822d-41f4-85e3-6865aee5bb05)

To predict land-use
Download the maps from the Swedish Land Survey (https://www.lantmateriet.se/sv/geodata/vara-produkter/produktlista/ekonomiska-kartan/). In all scripts, change the file paths to where you stored the data.

Run script 1CopyAllFilesToOneDir.py: Copy all tif files to one folder where you iterate over the tif files for the rest of the calculations. 

Run script 2DecompressTifs.sh: This is a bash script that converts the maps to a decompressed format that Whitebox can read.

Scripts 3-15 extract features from the maps:

Run script 3SplitCompositetoRGB.py: This script takes the original scanned image with three bands and saves it out to separate bands (Red, Green and Blue)

Run script 4RGBtoIHS.py: This converts the RGB files to Intensity, Hue and Saturation.

Run Script 5B_R.py: This divides the Blue band by Red band 

Run Script 6G_R.py: This divides the Green band by Red band 

Run Script 7B_G.py: This divides the Blue band by Green band
 
Run script 8MinFilter3cells.py: This runs a minimum filter on the Green band with a kernel of 3 cells.

Run script 9MedianFilterR19cells.py: This runs a median filter calculated on the Red band with a kernel of 19 cells

Run script 10GaussianFRsigma10.py: This runs a Gaussian filter calculated on the Red band with a sigma of 10

Run script 11StDevFR19cells.py: This runs a Standard Deviation filter calculated on Red band with a kernel of 19 cells

Run script 12StDevFI49cells.py: This runs a Standard Deviation filter calculated on Intensity with a kernel of 49 cells


Run script 13MapAverageG.py: This script calculates the average of the Green band for the entire map

Run script 14MapAverageHue.py: This script calculates the average of the Hue band for the entire map

Run script 15MapAverageSaturation: This script calculates the average of the Saturation band for the entire map


Now it’s time to load the pre-trained model and start running predictions: First, remove the folders of variables not included in the final model ('Red', 'Green', 'Blue', 'Intensity', 'Saturation', 'AverageR', 'AverageB', 'AverageI', 'StDevFI11') from the path to the input data. 

Run Script 16PredictionsFromXGBmodell.py which classifies the land use based on the input from the final 12 features. 

Run Script 17FixWater.py to burn Water in from modern maps. (Download Fastighetskartan from the Swedish Land Survey).

Run Script 18PostProcessing.py This script removes smaller “clumps” than 9 cells, in our-post-processing step. (This script I ran on Windows instead of Linux, as I had issues with activating the Whitebox extension on my Linux server. Hence search paths are written differently. Modify to your needs.)

Run 19SaveAs8BitUnsigned.py, to save some space. The size of each tif-file decreases from ca 50 000kb to ca 25 000kb.

Run Script 20MosaicMaps.sh which mosaics the maps into LandUse for North, Middle, and South, and Northern Inland of Sweden.

Finally we provide the two scripts for masking away graphics from the maps (21 Nodata.py and 22MaskGraphics).

Run Script 23MosaicMapsNoGraphics.sh which mosaics the maps into LandUse for North, Middle, and South, and Northern Inland of Sweden.


In addition, we provide the script where we trained the final XGB model (XGboostModelFinal.py + the pickled file it refers to (xgboost_model20.pkl)), and the excel file with the extracted features to the classified points (XGBpoints.xlsx).

We provide 2 scripts for automatically cropping and georeferencing the maps for the Northern Inland (1:20 000) (CropNorhthernInland.py and GeoreferenceNorhthernInland.py). 


