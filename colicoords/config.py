class cfg(object):

    #Pixel sizes are in nm, displayed units are um
    STORM_PIXELSIZE = 16 # deprecated
    IMG_PIXELSIZE = 80
    PAD_WIDTH = 3
    CELL_FRACTION = 0.5


    #Optimization bounds defaults
    ENDCAP_RANGE = 20

    #plotting parameters

    #Distribution plotting binsizes
    R_DIST_STOP = 30
    R_DIST_STEP = 0.5
    R_DIST_NORM_STOP = 2
    R_DIST_NORM_STEP = 0.1
    L_DIST_NBINS = 20
    #alpha dist?
    ALHPA_DIST_STOP = 180
    ALPHA_DIST_STEP = 1