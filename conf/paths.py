import os

PREPARED_PATH = 'prepared'

OPT_PATH = os.path.join('D:','Ferrari', 'Projects', 'trans_fusion', 'img', 'opt')
SAR_PATH = os.path.join('D:','Ferrari', 'Projects', 'trans_fusion', 'img', 'sar')
GENERAL_PATH = os.path.join('D:','Ferrari', 'Projects', 'trans_fusion', 'img', 'general')

EXPERIMENTS_FOLDER = os.path.join('D:','Ferrari', 'Projects', 'trans_fusion', 'exps')

OPT_IMGS = ['18193.tif', '18208.tif', '18218.tif', '19208.tif', '19228.tif', '19233.tif', '20213.tif', '20218.tif', '20223.tif']

SAR_IMGS = ['18198.tif', '18210.tif', '18222.tif', '19217.tif', '19229.tif', '19241.tif', '20212.tif', '20218.tif', '20224.tif']

OPT_TIFF_FILE_0 = os.path.join(OPT_PATH, OPT_IMGS[0])
TILES_PATH = os.path.join(GENERAL_PATH, 'tiles.tif')

SHP_FOLDER = os.path.join('D:','Ferrari', 'Projects', 'trans_fusion', 'shp')
PRODES_YEAR_DEF_SHP = os.path.join(SHP_FOLDER, 'yearly_deforestation.shp')
PRODES_PREV_DEF_SHP = os.path.join(SHP_FOLDER, 'accumulated_deforestation_2007.shp')
PRODES_HYDRO_SHP = os.path.join(SHP_FOLDER, 'hydrography.shp')
PRODES_NO_FOREST_DEF_SHP = os.path.join(SHP_FOLDER, 'no_forest.shp')
PRODES_RESIDUAL_SHP = os.path.join(SHP_FOLDER, 'residual.shp')