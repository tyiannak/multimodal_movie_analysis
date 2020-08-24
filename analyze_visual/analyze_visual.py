import cv2, time, sys, glob, os
import numpy
import scipy.cluster.hierarchy as hier
import scipy.spatial.distance as dist
import collections

# process and plot related parameters:
newWidth = 500; processStep = 0.5; plotStep = 2;

# face detection-related paths:
HAAR_CASCADE_PATH_FRONTAL = "haarcascade_frontalface_default.xml"
HAAR_CASCADE_PATH_PROFILE = "haarcascade_frontalface_default.xml"

# flow-related parameters:
lk_params = dict( winSize  = (15, 15), 
                  maxLevel = 5, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                              10, 0.03))
feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 3,
                       blockSize = 5 )


def angleDiff(angle1, angle2):
    # angles (in degs) difference
    Diff = numpy.abs(angle2-angle1);
    if numpy.abs(Diff) > 180:
        Diff = 360 - Diff
    return Diff


def anglesSTD(angles, MEAN):
    # computes the standard deviation between a set of angles
    S = 0.0;
    for a in angles:
        S += (angleDiff(a, MEAN)**2)
    S /= len(angles)
    S = numpy.sqrt(S)
    return S


def anglesCluster(angles):
    # this function uses hierarchical clustering to find the
    # clusters of a set of angle values
    Dists = dist.pdist(angles, angleDiff)
    linkageMatrix = hier.linkage(Dists, metric = angleDiff)
    C = hier.fcluster(linkageMatrix, 5, 'maxclust')

    return C
    

def drawArrow(image, p, q, color, arrowMagnitude = 5, thickness=1, line_type=8,
              shift=0):
    # Draw the principle line
    cv2.line(image, tuple(p), tuple(q), color, thickness, line_type, shift);
    # compute the angle alpha
    angle = numpy.arctan2( p[1]-q[1], p[0]-q[0]);
    # compute the coordinates of the first segment
    p[0] =  int(q[0] +  arrowMagnitude * numpy.cos(angle + numpy.pi/4.0));
    p[1] =  int(q[1] +  arrowMagnitude * numpy.sin(angle + numpy.pi/4.0));
    # /Draw the first segment
    cv2.line(image, tuple(p), tuple(q), color, thickness, line_type, shift);
    # compute the coordinates of the second segment
    p[0] =  int(q[0] +  arrowMagnitude * numpy.cos(angle - numpy.pi/4.0));
    p[1] =  int(q[1] +  arrowMagnitude * numpy.sin(angle - numpy.pi/4.0));

    # Draw the second segment
    cv2.line(image, tuple(p), tuple(q), color, thickness, line_type, shift);
    return image


def plotCV(Fun, Width, Height, MAX):
    if len(Fun)>Width:
        hist_item = Height * (Fun[len(Fun)-Width-1:-1] / MAX)
    else:
        hist_item = Height * (Fun / MAX)
    h = numpy.zeros((Height, Width, 3))
    hist = numpy.int32(numpy.around(hist_item))

    for x,y in enumerate(hist):
        cv2.line(h,(x,Height),(x,Height-y),(255,0,255))
    return h

def intersect_rectangles(r1, r2):
    x11 = r1[0]; y11 = r1[1]; x12 = r1[0]+r1[2]; y12 = r1[1]+r1[3];
    x21 = r2[0]; y21 = r2[1]; x22 = r2[0]+r2[2]; y22 = r2[1]+r2[3];
        
    X1 = max(x11, x21); X2 = min(x12, x22);
    Y1 = max(y11, y21); Y2 = min(y12, y22);

    W = X2 - X1
    H = Y2 - Y1
    if (H>0) and (W>0):
        E = W * H;
    else:
        E = 0.0;
    Eratio = 2.0*E / (r1[2]*r1[3] + r2[2]*r2[3])
    return Eratio

def initialize_face():
    cascadeFrontal = cv2.CascadeClassifier(HAAR_CASCADE_PATH_FRONTAL);
    cascadeProfile = cv2.CascadeClassifier(HAAR_CASCADE_PATH_PROFILE);
    return (cascadeFrontal, cascadeProfile)

def detect_faces(image, cascadeFrontal, cascadeProfile):
    facesFrontal = []; facesProfile = []
    detectedFrontal = cascadeFrontal.detectMultiScale(image, 1.3, 5)
    print(detectedFrontal)
    if len(detectedFrontal)>0:
        for (x,y,w,h) in detectedFrontal:
            facesFrontal.append((x,y,w,h))
    #if detectedProfile:
    #    for (x,y,w,h),n in detectedProfile:
    #        facesProfile.append((x,y,w,h))

    # remove overlaps:
    while (1):
        Found = False
        for i in range(len(facesFrontal)):
            for j in range(len(facesFrontal)):
                if i != j:
                    interRatio = intersect_rectangles(facesFrontal[i],
                                                      facesFrontal[j])
                    if interRatio>0.3:
                        Found = True;
                        del facesFrontal[i]
                        break;
            if Found:
                break;

        if not Found:    # not a single overlap has been detected -> exit loop
            break;


    #return (facesFrontal, facesProfile)
    return (facesFrontal)

def resizeFrame(frame, targetWidth):    
    (Width, Height) = frame.shape[1], frame.shape[0]

    if targetWidth > 0:  # Use FrameWidth = 0 for NO frame resizing
        ratio = float(Width) / targetWidth        
        newHeight = int(round(float(Height) / ratio))
        frameFinal = cv2.resize(frame, (targetWidth, newHeight))
    else:
        frameFinal = frame;

    return frameFinal
    
def getRGBHistograms(RGBimage):
    # compute histograms:
    [histR, bin_edges] = numpy.histogram(RGBimage[:,:,0], bins=range(-1,256,32))
    [histG, bin_edges] = numpy.histogram(RGBimage[:,:,1], bins=range(-1,256,32))
    [histB, bin_edges] = numpy.histogram(RGBimage[:,:,2], bins=range(-1,256,32))
    # normalize histograms:
    histR = histR.astype(float); histR = histR / numpy.sum(histR);
    histG = histG.astype(float); histG = histG / numpy.sum(histG);
    histB = histB.astype(float); histB = histB / numpy.sum(histB);
    return (histR, histG, histB)

def getHSVHistograms(HSVimage):
    # compute histograms:
    [histH, bin_edges] = numpy.histogram(HSVimage[:,:,0], bins=range(180))
    [histS, bin_edges] = numpy.histogram(HSVimage[:,:,1], bins=range(256))
    [histV, bin_edges] = numpy.histogram(HSVimage[:,:,2], bins=range(256))
    # normalize histograms:
    histH = histH.astype(float); histH = histH / numpy.sum(histH);
    histS = histS.astype(float); histS = histS / numpy.sum(histS);
    histV = histV.astype(float); histV = histV / numpy.sum(histV);
    return (histH, histS, histV)

def getHSHistograms_2D(HSVimage):
    (Width, Height) = HSVimage.shape[1], HSVimage.shape[0]    
    H, xedges, yedges = numpy.histogram2d(numpy.reshape(HSVimage[:,:,0],
                                                        Width*Height),
                                          numpy.reshape(HSVimage[:,:,1],
                                                        Width*Height),
                                          bins=(range(-1,180, 30),
                                                range(-1, 256, 64)))
    H = H / numpy.sum(H);
    return (H, xedges, yedges)

def computeFlowFeatures(Grayscale, GrayscalePrev, p0, lk_params):
    p1, st, err = cv2.calcOpticalFlowPyrLK(GrayscalePrev, Grayscale, p0,
                                           None, **lk_params)
    good_new = p1[st==1]
    good_old = p0[st==1]
    angles = []
    mags = []
    dxS = []; dyS = []
    # draw motion arrows
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        x1, y1 = new.ravel(); x2, y2 = old.ravel()
        dx = x2 - x1; dy = y2 - y1
        if dy < 0:
            angles.append([numpy.abs(180.0 * numpy.arctan2( dy, dx) / numpy.pi)])
        else:
            angles.append([360.0 - 180.0 * numpy.arctan2( dy, dx) / numpy.pi])
        mags.append(numpy.sqrt(dx*dx + dy*dy) /
                    numpy.sqrt(Grayscale.shape[0]*Grayscale.shape[0] +
                               Grayscale.shape[1]*Grayscale.shape[1]))
        dxS.append(dx); dyS.append(dy);
    angles = numpy.array(angles);
    mags = numpy.array(mags);
    DistHorizontal = -1; 
    if len(angles)>0:
        meanDx = numpy.mean(dxS); meanDy = numpy.mean(dyS);
        if meanDy < 0:
            MEANANGLE = -(180.0 * numpy.arctan2( int(meanDy),
                                                 int(meanDx)) / numpy.pi)
        else:
            MEANANGLE = 360.0 - (180.0 * numpy.arctan2( int(meanDy),
                                                        int(meanDx)) / numpy.pi)
        STD = anglesSTD(angles, MEANANGLE)

        DistHorizontal = min(angleDiff(MEANANGLE, 180), angleDiff(MEANANGLE, 0))
        TitlPanConfidence = numpy.mean(mags) / numpy.sqrt(STD + 0.000000010)
        TitlPanConfidence = TitlPanConfidence[0]
        # TODO: 
        # CHECK PANCONFIDENCE
        # SAME FOR ZOOM AND OTHER CAMERA EFFECTS
        if TitlPanConfidence < 1.0:
            TitlPanConfidence = 0;
            DistHorizontal = -1;
    else:
        mags = [0];
        angles = [0];
        dxS = [0];
        dyS = [0];
        MEANANGLE = 0
        STD = 0
        TitlPanConfidence = 0.0

    return (angles, mags, MEANANGLE, STD, good_new, good_old, dxS, dyS, TitlPanConfidence)


def processMovie(moviePath, processMode, PLOT):
    Tstart = time.time(); T0 = Tstart;
    capture = cv2.VideoCapture(moviePath)
    nFrames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = capture.get(cv2.CAP_PROP_FPS)
    duration = nFrames / fps
    secondsD  = duration; HoursD   = int(secondsD/3600)
    MinutesD = int(secondsD/60)
    SecondsD = int(secondsD) % 60; DsecsD = int(100*(secondsD - int(secondsD)))
    StringTimeD = '{0:02d}:{1:02d}:{2:02d}.{3:02d}'.format(HoursD, MinutesD,
                                                           SecondsD, DsecsD)
    if PLOT:
        print("FPS      = " + str(fps))
        print("Duration = " + str(duration) + " - " + StringTimeD)    

    pOld = numpy.array([])
    timeStamps = numpy.array([])
    frame_val_dif = numpy.array([])
    flowAngle = numpy.array([])
    flowMag   = numpy.array([])
    flowStd   = numpy.array([])
    processFPS = numpy.array([])
    processT   = numpy.array([])
    
    if processMode > 1:
        NFacesFrontal = collections.deque(maxlen= 200)
        PFacesFrontal = collections.deque(maxlen= 200)
        TitlPanConfidences = collections.deque(maxlen= 200)
    count = 0
    countProcess = 0

    if processMode>1:
        (cascadeFrontal, cascadeProfile) = initialize_face()

    nextTimeStampToProcess = 0.0
    PROCESS_NOW = False
    shotChangeTimes = [0]
    shotChangeProcessIndices = [0]
    shotDurations = []

    while (1):
        # cv.SetCaptureProperty( capture, cv.CV_CAP_PROP_POS_FRAMES,
        # count*frameStep );
        # (THIS IS TOOOOO SLOW (MAKES THE READING PROCESS 2xSLOWER))

        ret, frame = capture.read()
        timeStamp = float(count) / fps
        if timeStamp >= nextTimeStampToProcess:
            nextTimeStampToProcess += processStep;
            PROCESS_NOW = True
        if ret:
            count += 1; 
            (Width, Height) = frame.shape[1], frame.shape[0]
            frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            RGB = resizeFrame(frame2, newWidth)
            Grayscale = cv2.cvtColor(RGB, cv2.COLOR_RGB2GRAY)
            (Width, Height) = Grayscale.shape[1], Grayscale.shape[0]

            if processMode>1:
                if (count % 25) == 1:    
                    p0 = cv2.goodFeaturesToTrack(Grayscale, mask = None,
                                                 **feature_params)
                    if (p0 is None):
                        p0 = pOld;
                    pOld = p0;

            if PROCESS_NOW:
                curFV = numpy.array([])

                countProcess += 1
                timeStamps = numpy.append(timeStamps, timeStamp)
            
                if processMode>0:
                    #[histH, histS, histV] = getHSVHistograms(HSV)
                    # PROCESS LEVEL 1:
                    HSV = cv2.cvtColor(RGB, cv2.COLOR_RGB2HSV)
                    #histHS, xedges, yedges = getHSHistograms_2D(HSV)

                    [histR, histG, histB] = getRGBHistograms(RGB)

                    RGBratio = 100.0* (numpy.max(RGB, 2) -
                                       numpy.mean(RGB, 2)) / \
                               (1.0+numpy.mean(RGB, 2))
                    Vnorm    = (255.0 * HSV[:,:,2]) / numpy.max(HSV[:,:,2]+1.0)                    
                    Snorm    = (255.0 * HSV[:,:,1]) / numpy.max(HSV[:,:,1]+1.0)                    

                    RGBratio[RGBratio>199.0] = 199.0;
                    RGBratio[RGBratio<1.0] = 1.0;
                    histRGBratio, _ = numpy.histogram(RGBratio, 
                                                      bins=range(-1,200,40))
                    histRGBratio = histRGBratio.astype(float)
                    histRGBratio = histRGBratio / numpy.sum(histRGBratio)
                    histV, _ = numpy.histogram(Vnorm, bins=range(-1,256,32))
                    histV = histV.astype(float)
                    histV = histV / numpy.sum(histV)
                    histS, _ = numpy.histogram(Snorm, bins=range(-1,256,32))
                    histS = histS.astype(float)
                    histS = histS / numpy.sum(histS)

                    # update the current feature vector
                    curFV = numpy.concatenate((curFV, histR), 0)  
                    curFV = numpy.concatenate((curFV, histG), 0)  
                    curFV = numpy.concatenate((curFV, histB), 0)  
                    curFV = numpy.concatenate((curFV, histV), 0)            
                    curFV = numpy.concatenate((curFV, histRGBratio), 0)                                
                    curFV = numpy.concatenate((curFV, histS), 0)                
                    #curFV = numpy.concatenate((curFV, numpy.reshape(histHS, 
                    # histHS.shape[0]*histHS.shape[1])), 1)
                    if countProcess>1:
                        frame_val_dif = numpy.append(frame_val_dif,
                                                     numpy.mean(numpy.mean(numpy.abs(histV - histVprev))))
                    else:
                        frame_val_dif = numpy.append(frame_val_dif, 0.0);

                    curFV = numpy.concatenate((curFV, numpy.array([frame_val_dif[-1]])), 0)                    

                    #histHSprev = histHS;
                    histVprev = histV;

#                    print str(len(histV)) + ' value hist  ' + str(histHS.shape[0]*histHS.shape[1]) + ' HS 2D hist ' ,

                if processMode > 1:
                    facesFrontal = detect_faces(RGB, cascadeFrontal, cascadeProfile) # face detection                    
                    NFacesFrontal.append(float(len(facesFrontal)))                                              # update number of faces
                    if len(facesFrontal)>0:
                        tempF = 0.0
                        for f in facesFrontal:
                            tempF += (f[2] * f[3] / float(Width * Height))                                      # face size ratio (normalzied to the frame dimensions)                        
                        PFacesFrontal.append(tempF/len(facesFrontal))                                           # average "face ratio"
                    else:
                        PFacesFrontal.append(0.0)
                    if countProcess>1 and len(p0)>0:
                        angles, mags, MEANANGLE, STD, good_new, good_old, dxS, dyS, TitlPanConfidence = computeFlowFeatures(Grayscale, GrayscalePrev, p0, lk_params)
                        meanMag = numpy.mean(numpy.array(mags))
                        stdMag = numpy.std(numpy.array(mags))
                        TitlPanConfidences.append(TitlPanConfidence)
                    else:
                        TitlPanConfidences.append(0.0)
                        meanMag = 0                    
                        stdMag = 0                    
                    if countProcess > 1:
                        grayDiff = (GrayscalePrev - Grayscale)
                        grayDiff[grayDiff<50] = 0
                        grayDiff[grayDiff>50] = 1
                        grayDiffT = grayDiff.sum() / float(grayDiff.shape[0] * grayDiff.shape[1])                        
                        #print "{0:.3f}\t{1:.3f}\t{2:.3f}".format(meanMag, grayDiffT, frame_val_dif[-1]),
                        if (meanMag > 0.06) and (grayDiffT > 0.55) and (frame_val_dif[-1] > 0.002):                                       # shot change detection                        
                            if timeStamp - shotChangeTimes[-1] > 1.1:
                                averageShot = 0
                                if len(shotChangeTimes)-1 > 5:
                                    for si in range(len(shotChangeTimes)-1):
                                        averageShot += (shotChangeTimes[si+1]-shotChangeTimes[si])
                                    print(averageShot / float(len(shotChangeTimes)-1))

                                shotChangeTimes.append(timeStamp)
                                shotChangeProcessIndices.append(countProcess)
                                for ccc in range(shotChangeProcessIndices[-1] - shotChangeProcessIndices[-2]):
                                    shotDurations.append(shotChangeProcessIndices[-1] - shotChangeProcessIndices[-2])
                                #print "FOUND"
                        #else:
                        #    print ""

                    curFV = numpy.concatenate((curFV, numpy.array([NFacesFrontal[-1]])), 0)
                    curFV = numpy.concatenate((curFV, numpy.array([PFacesFrontal[-1]])), 0)
                    curFV = numpy.concatenate((curFV, numpy.array([TitlPanConfidences[-1]])), 0)
                    curFV = numpy.concatenate((curFV, numpy.array([meanMag])), 0)
                    curFV = numpy.concatenate((curFV, numpy.array([stdMag])), 0)

                if processMode > 0:
                    if countProcess == 1:
                        FeatureMatrix = numpy.reshape(curFV, ( 1, len(curFV)))
                    else:
                        FeatureMatrix = numpy.concatenate((FeatureMatrix, numpy.reshape(curFV, ( 1, len(curFV)))), 0)                    
                if ((countProcess > 2) and (countProcess % plotStep ==0) and (PLOT==1)):
                    # draw RGB image and visualizations
                    vis = cv2.cvtColor(RGB, cv2.COLOR_RGB2BGR)
    
                    if processMode>1 and len(p0)>0:                
                        # faces bounding boxes:
                        for f in facesFrontal:                            # draw face rectangles
                            cv2.rectangle(vis, (f[0], f[1]), (f[0]+f[2],f[1]+f[3]), (0,255,255), 3)
                        # flow arrows:
                        for i,(new,old) in enumerate(zip(good_new,good_old)):            # draw motion arrows
                            x1, y1 = new.ravel(); x2, y2 = old.ravel()
                            pt1 = [int(x1), int(y1)]; pt2 = [int(x2), int(y2)]
                            vis = drawArrow(vis, pt1, pt2, (0, 255, 0))

                        if len(angles)>0:
                            vis = drawArrow(vis, [int(Width/2), int(Height/2)], [int(Width/2)+int(numpy.mean(dxS)), int(Height/2)+int(numpy.mean(dyS))], (0, 0, 255), arrowMagnitude = 5, thickness=4, line_type=8, shift=0)                        
                        cv2.putText(vis,str(int(MEANANGLE)), (int(Width/2), int(Height/2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, 255)

                    # Time-related plots:
                    T2 = time.time();                
                    seconds  = float(count)/fps; Hours   = int(seconds/3600); Minutes = int(seconds/60); Seconds = int(seconds) % 60; Dsecs = int(100*(seconds - int(seconds)));
                    StringTime = '{0:02d}:{1:02d}:{2:02d}.{3:02d}'.format(Hours, Minutes, Seconds, Dsecs);
                    processFPS = numpy.append(processFPS, plotStep / float(T2-T0))
                    processT   = numpy.append(processT,   100.0 * float(T2-T0) / (processStep * plotStep))
                    if len(processFPS)>250:
                        processFPS_winaveg = numpy.mean(processFPS[-250:-1])
                        processT_winaveg = numpy.mean(processT[-250:-1])
                    else:
                        processFPS_winaveg = numpy.mean(processFPS)
                        processT_winaveg = numpy.mean(processT)

                    secondsRemain = processT_winaveg * float(secondsD - seconds) / 100.0; HoursRemain   = int(secondsRemain/3600); MinutesRemain = int(secondsRemain/60); SecondsRemain = int(secondsRemain) % 60; 

                    StringTimeRemain = '{0:02d}:{1:02d}:{2:02d}'.format(HoursRemain, MinutesRemain, SecondsRemain);
                    StringToPlot = '{0:s}/{1:s} {2:5.1f}fps,{3:2.1f}xR {4:s}'.format(StringTime, StringTimeD, processFPS_winaveg, 100.0/float(processT_winaveg),StringTimeRemain)                    
                    cv2.rectangle(vis, (0, 0), (Width, 17), (255,255,255), -1)
                    cv2.putText(vis, StringToPlot, (20, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0))

                    # Draw color image:
                    WidthPlot = 150;
                    WidthPlot2 = 150;            # used for static plot (e.g. 1D histogram)
                    cv2.imshow('Color', vis)
                    cv2.imshow('GrayNorm', Vnorm/256.0)
                    cv2.moveWindow('Color', 0, 0)
                    cv2.moveWindow('GrayNorm', newWidth , 0)

                    if processMode>0:
                        #histHSplot = (histHS / numpy.max(histHS))
                        #cv2.imshow('Hue-Saturation Hist', cv2.resize(histHSplot, (Height, Height), interpolation = cv2.INTER_CUBIC))        
                        #h = plotCV(frame_val_dif, WidthPlot, Height, 0.020);     cv2.imshow('HSV Diff',h)        
                        #h = plotCV(scipy.signal.resample(histRGBratio, 256), WidthPlot2, Height, numpy.max(histRGBratio)); cv2.imshow('Color Hist', h)                                
                        h = plotCV(numpy.repeat(histRGBratio, WidthPlot2 / histRGBratio.shape[0]), WidthPlot2, Height, numpy.max(histRGBratio)); cv2.imshow('Color Hist', h)        
                        h = plotCV(numpy.repeat(histV, WidthPlot2 / histV.shape[0]), WidthPlot2, Height, numpy.max(histV)); cv2.imshow('Value Hist', h)
                        h = plotCV(numpy.repeat(histS, WidthPlot2 / histS.shape[0]), WidthPlot2, Height, numpy.max(histS)); cv2.imshow('Sat Hist', h)
                        #cv2.moveWindow('Hue-Saturation Hist',     Width+50, 0)
                        cv2.moveWindow('Color Hist',   0,                Height + 70) 
                        cv2.moveWindow('Value Hist',   WidthPlot2 ,      Height + 70)
                        cv2.moveWindow('HSV Diff',     2 * WidthPlot2 ,  Height + 70)
                        cv2.moveWindow('Sat Hist',     2 * WidthPlot2 ,  Height + 70)
                    if processMode>1:                        
                        h = plotCV(numpy.array(NFacesFrontal), WidthPlot, Height, 5);cv2.imshow('NFacesFrontal', h)
                        h = plotCV(numpy.array(PFacesFrontal), WidthPlot, Height, 1);cv2.imshow('PFacesFrontal', h)
                        h = plotCV(numpy.array(TitlPanConfidences), WidthPlot, Height, 50);     cv2.imshow('TitlPanConfidences', h)
                        cv2.moveWindow('NFacesFrontal',         0,              2 * Height+70)
                        cv2.moveWindow('PFacesFrontal',         WidthPlot2,     2 * Height+70)
                        cv2.moveWindow('TitlPanConfidences',    2 * WidthPlot2, 2 * Height+70)
                    ch = cv2.waitKey(1)
                    T0 = T2;
                PROCESS_NOW = False
                GrayscalePrev = Grayscale;
                #print FeatureMatrix.shape

        else:
            break;
    
    if processMode > 0:
        numpy.savetxt("features.csv", FeatureMatrix, delimiter=",")

    processingTime = time.time() - Tstart
    processingFPS = countProcess / float(processingTime); 
    processingRT = 100.0 * float(processingTime) / (duration);

    seconds  = processingTime; Hours   = int(seconds/3600); Minutes = int(seconds/60); Seconds = int(seconds) % 60; Dsecs = int(100*(seconds - Seconds));

    if PLOT:
        print("processing time: " + '{0:02d}:{1:02d}:{2:02d}.{3:02d}'.format(Hours, Minutes, Seconds, Dsecs));
        print("processing ratio      {0:3.1f} fps".format(processingFPS))
        print("processing ratio time {0:3.0f} %".format(processingRT))

    for ccc in range(countProcess - shotChangeProcessIndices[-1]):
        shotDurations.append(countProcess - shotChangeProcessIndices[-1])    

    shotDurations = numpy.matrix(shotDurations)
    shotDurations = shotDurations * processStep
    #print shotDurations
    #print shotDurations.shape
    #print shotChangeTimes
    FeatureMatrix = numpy.append(FeatureMatrix, shotDurations.T, axis = 1)

    Fm  = FeatureMatrix.mean(axis=0)
    Fs  = FeatureMatrix.std(axis=0)
    Fsm = FeatureMatrix.std(axis=0) / (numpy.median(FeatureMatrix, axis=0) + 0.0001)
    FeatureMatrixSortedRows = numpy.sort(FeatureMatrix, axis=0)
    FeatureMatrixSortedRowsTop10 = FeatureMatrixSortedRows[ - int(0.10 * FeatureMatrixSortedRows.shape[0])::, :]
    Fm10top = FeatureMatrixSortedRowsTop10.mean(axis=0)    
    F = numpy.concatenate((Fm, Fs, Fsm, Fm10top), axis = 1)
    print(FeatureMatrix.shape)
    #print Fm.shape, Fs.shape, Fsm.shape, Fm10top.shape
    print(F.shape    )
        
    return F, FeatureMatrix

def dirProcessMovie(dirName):
    """
    """
    dirNameNoPath = os.path.basename(os.path.normpath(dirName))

    allFeatures = numpy.array([])

    types = ('*.avi', '*.mpeg',  '*.mpg', '*.mp4', '*.mkv')
    movieFilesList = []
    for files in types:
        movieFilesList.extend(glob.glob(os.path.join(dirName, files)))    
    movieFilesList = sorted(movieFilesList)
    
    for movieFile in movieFilesList:    
        print(movieFile)
        [F, FeatureMatrix] = processMovie(movieFile, 2, 1)
        numpy.save(movieFile+".npy", FeatureMatrix)
        if len(allFeatures)==0:                # append feature vector
            allFeatures = F            
        else:
            allFeatures = numpy.vstack((allFeatures, F))
    numpy.save(dirNameNoPath + "_features.npy", allFeatures)            
    numpy.save(dirNameNoPath + "_movieFilesList.npy", movieFilesList)            
    return allFeatures, movieFilesList

def dirsProcessMovie(dirNames):
    # feature extraction for each class:
    features = [];
    classNames = []
    fileNames = []
    for i,d in enumerate(dirNames):
        [f, fn] = dirProcessMovie(d)
        if f.shape[0] > 0: # if at least one audio file has been found in the provided folder:
            features.append(f)
            fileNames.append(fn)
            if d[-1] == "/":
                classNames.append(d.split(os.sep)[-2])
            else:
                classNames.append(d.split(os.sep)[-1])
    return features, classNames, fileNames

def npyToCSV(fileNameFeatures, fileNameNames):
    F = numpy.load(fileNameFeatures)
    N = numpy.load(fileNameNames)
    fp = open(fileNameFeatures.replace(".npy",".csv"), 'w')
    for i in range(len(N)):
        fp.write(os.path.basename(os.path.normpath(N[i])) + "\t"), 
        for f in F[i]:
            fp.write("{0:.6f}\t".format(f))
        fp.write("\n")

def analyze(fileNameFeatures, fileNameNames, startF = 0, endF = 108, particularFeatures = []):
    f = 0
    F = numpy.load(fileNameFeatures)
    N = numpy.load(fileNameNames)

    text_file = open("ground_names.txt", "r")
    gtNames = lines = text_file.readlines();
    gtNames = [g.replace("\n","") for g in gtNames]
    gtSim = numpy.load("ground_sim_numpy")    

    # normalize    
    MEAN = numpy.mean(F, axis = 0); STD  = numpy.std(F, axis = 0)    
    for i in range(F.shape[0]):
        F[i,:] = (F[i] - MEAN) / STD    
    
    firstPos = []
    secondPos = []
    top10 = []
    top10_second = []

    for i in range(len(N)):         # for each movie                
        curName = os.path.basename(os.path.normpath(N[i])).replace(".mkv","").replace(".mpg","").replace(".mp4","").replace(".avi","")
        gtIndex = gtNames.index(curName)
        curGTSim = gtSim[gtIndex, :]
        curGTSim[gtIndex] = 0
        iGTmin = numpy.argmax(curGTSim)
        gtSorted = [x for (y,x) in sorted(zip(curGTSim, gtNames), reverse=True)]
        #print curName
        #for c in range(10):
        #    print "   " + gtSorted[c]        

        featuresToUse = range(startF, endF)
        if len(particularFeatures) > 0:
            featuresToUse = particularFeatures
        else:
            featuresToUse = range(startF,endF)

        #featuresToUse = [5, 6, 7, 13, 16, 17, 25, 26, 27, 32, 34, 41, 42, 43, 44, 47]
        #featuresToUse = [0,1, 3, 4, 5, 6, 7, 8, 10,11,12,13,14,15, 16, 17, 25, 28, 35, 42, 43, 44, 48, 53, 59, 66, 71, 78, 94, 95, 97, 98, 100, 101, 102]
        #featuresToUse = [106, 12, 36, 51, 3, 2, 89, 93, 65, 16, 96, 76, 25, 80, 20, 79, 72, 7, 60, 44]
        F[:,featuresToUse].shape
        d = dist.cdist(F[i, featuresToUse].reshape(1, len(featuresToUse)), F[:,featuresToUse])
        d[0][i] = 100000000
        d = d.flatten()
        #print d.shape, len(N)
        rSorted = [os.path.basename(os.path.normpath(x)).replace(".mkv","").replace(".mpg","").replace(".mp4","").replace(".avi","") for (y,x) in sorted(zip(d.tolist(), N))]        

        firstPos.append(gtSorted.index(rSorted[0]) + 1)
        secondPos.append(gtSorted.index(rSorted[1]) + 1)
        if rSorted[0] in gtSorted[0:10]:
            top10.append(1)
        else:
            top10.append(0)
        if rSorted[1] in gtSorted[0:10]:
            top10_second.append(1)
        else:
            top10_second.append(0)

        #print rSorted
        #print curName
        #for c in range(3):
        #    print  "         " + rSorted[c]        
        #print "{0:60s}\t{1:60s}".format( os.path.basename(os.path.normpath(N[i])), os.path.basename(os.path.normpath(N[numpy.argmin(d)])))
    #print numpy.median(numpy.array(firstPos)), 100*numpy.sum(numpy.array(top10)) / len(top10)
    return numpy.median(numpy.array(firstPos)), 100*numpy.sum(numpy.array(top10)) / len(top10), numpy.median(numpy.array(secondPos)), 100*numpy.sum(numpy.array(top10_second)) / len(top10_second)

def scriptAnalyze():
    nExp = 1000
    allFeatureCombinations = []
    medPos = []
    top10 = []
    medPos2 = []
    top102 = []

    T1 = 50
    T2 = 20
    for nFeatures in [5, 10, 20, 30, 40, 50, 60, 70]:
        print(nFeatures)
        for e in range(nExp):
            curFeatures = numpy.random.permutation(range(108))[0:nFeatures]
            allFeatureCombinations.append(curFeatures)
            a1, a2, a3, a4 = analyze("featuresAll.npy", "namesAll.npy", 0, 0, curFeatures)
            medPos.append(a1)
            top10.append(a2)
            medPos2.append(a3)
            top102.append(a4)

    medPos = numpy.array(medPos)
    top10 = numpy.array(top10)    
    medPos2 = numpy.array(medPos2)
    top102 = numpy.array(top102)    

    iMinPos = numpy.argmin(medPos)
    iMaxPos = numpy.argmax(top10)
    iMinPos2 = numpy.argmin(medPos2)
    iMaxPos2 = numpy.argmax(top102)

    for i in range(len(top10)):
        if (medPos[i] < T1) and (top10[i] > T2) and (medPos2[i] < T1) and (top102[i] > T2):
            print("{0:.1f}\t{1:.1f}\t{2:.1f}\t{3:.1f}".format(medPos[i], top10[i], medPos2[i], top102[i]))
            if i == iMinPos:
                print("BEST medPos\t")
            else:
                print("-----------\t")
            if i == iMaxPos:
                print("BEST top10\t")
            else:
                print("----------\t")
            if i == iMinPos2:
                print("BEST medPos2\t")
            else:
                print("------------\t")
            if i == iMaxPos2:
                print("BEST top102\t")
            else:
                print("-----------\t")

            for f in allFeatureCombinations[i]:
                print("{0:d},".format(f))         
            

def main(argv):
    if len(argv)==3:
        if argv[1]=="-f":  # single file
            processMovie(argv[2], 2, 1)
    if argv[1]=="-d":      # directory
        dirName = argv[2]
        allFeatures, movieFilesList = dirProcessMovie(dirName)
        print(allFeatures.shape, movieFilesList)
        #F = dirsProcessMovie(dirNames)
    if argv[1]=="evaluate":
        [a, b, a2, b2] = analyze("featuresAll.npy", "namesAll.npy")
        print("First returned result median position {0:.1f}".format(a))
        print("First returned result in top10 {0:.1f} %".format(b))
        print("Second returned result median position {0:.1f}".format(a2))
        print("Second returned result in top10 {0:.1f} %".format(b2))

    if argv[1]=="scriptDebug":         
        scriptAnalyze()

if __name__ == '__main__':
    main(sys.argv)

