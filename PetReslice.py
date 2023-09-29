"""
M. Palmer, November, 2022
PET volume reslicing/resampling script.
Motivated by ACR's requirement for a dataset resampled at 1 cm z-spacing.

Possible additions for future:
- use scipy.signal.decimate to downsample.  Not sure if there's anthing to gain beyond slabbing.

"""
import os, sys, time
import datetime
import numpy as np
import pydicom as dicom
from pydicom.uid import generate_uid
from pydicom.filereader import read_dicomdir
import scipy.ndimage as snd
import scipy.interpolate as si
import datetime as dt

from gui import gui_dicomdir

progname = os.path.basename(sys.argv[0])
__version__ = "0.9"


implementation_class_uid = dicom.uid.PYDICOM_IMPLEMENTATION_UID     #might be set automatically upon writing
media_storage_uid = generate_uid()
sop_instance_uid = media_storage_uid
implementation_version_name = progname + ' ' + __version__
source_application_entity_title = progname
series_instance_uid = generate_uid()

modification_time = time.strftime("%H%M%S")
modification_date = time.strftime("%Y%m%d")

DZ_TOL = 0.1        # Tolerance on z-sampling in mm for input data.
XY_TOL = 0.1        # origin can shift in x-y plane by only this much over the z extent.
SERIES_DESCRIPTION_PREFIX = 'Resample: '
SERIES_NUMBER_BIAS = 500
IMAGE_TYPE = "DERIVED\\SECONDARY"

OUTPUT_SLICE_SPACING = 10.0

MAX16 = 10000       # max and min 2's complement integers to make use of
MIN16 = 0
SUV_MAX_DISP = 3.5      # to set display center/level


def vol_qc(vol_ds):
    """
    Do some QC on the volume data.  ImagePositionPatient should have same x,y coordinates.  z should
    have uniformy-sampled data, no gaps. ImageOrientationPatient should be identical

    Parameters
    ----------
    vol_ds: list of DICOM datasets

    Returns
    -------

    vol_ds : ndarray
        z-axis coordinates
    rev : bool
        true if slices are ordered in reverse when they're sorted by increasing z

    """
    z = np.array([ds.ImagePositionPatient[2] for ds in vol_ds])
    dz = np.diff(z)
    x = [ds.ImagePositionPatient[0] for ds in vol_ds]
    y = [ds.ImagePositionPatient[1] for ds in vol_ds]

    if abs(np.max(dz) - np.min(dz)) > DZ_TOL:
        raise ValueError('dZ data gap or tolerance exceeded')
    if abs(np.max(x) - np.min(x)) > XY_TOL or abs(np.max(y) - np.min(y)) > XY_TOL:
        raise ValueError('x,y origin drift tolerance exceeded')

    for d in range(6):
        rng = np.max([ds.ImageOrientationPatient[d] for ds in vol_ds]) - np.min([ds.ImageOrientationPatient[d] for ds in vol_ds])
        if abs(rng) > 0.01:
            raise ValueError('Image orientation vectors in volume cannot chante')

    inum = [ds.get('InstanceNunber', 0) for ds in vol_ds]
    rev = inum[-1] < inum[0]

    # note that z is increasing

    #for ds in vol_ds:
    #    print(np.max(ds.pixel_array), ds.RescaleSlope, ds.RescaleIntercept)


    return z, rev

def get_input_filelist(indir):
    """
    Read DICOMs specified in a list of input filenames.  Return a list of pydicom datasets.
    Parameters
    ----------
    indir : str
        either dirname or DICOMDIR

    Returns
    -------
    filelist : list

    """
    filelist = []


    if os.path.isfile(indir):
        try:
            read_dicomdir(indir) # to force it to crap out if not DICOMDIR
            filelist = gui_dicomdir(indir)
        except:
            raise SystemError('Source is a file but not DICOMDIR')

        return filelist

    """
    Here because not DICOMDIR
    """

    for dirpath, dirnames, filenames in os.walk(indir):
        for f in filenames:
            filelist += [os.path.join(dirpath, f)]

    return filelist

def suv_scale(ds):
    """ Return the decay factor """

    def _float_or_none(c):
        try:
            r = float(c)
        except:
            r = None
        return r

    def _cleanT(str_dt):
        """
        Clean off the trailing ms field if it exists in a string date spec
        """
        #            parts = str_dt.split(".")
        #            return parts[0]
        return str_dt.split(".")[0]


    StrDTdicom = "%Y%m%d%H%M%S"  # DICOM way to storre Date/time
    LN2 = np.log(2)

    rpis = ds.get('RadiopharmaceuticalInformationSequence')
    if rpis is None:
        return None

    injdt = rpis[0].get('RadiopharmaceuticalStartDateTime')
    if injdt is not None:
        inj_datetime = dt.datetime.strptime(_cleanT(injdt), StrDTdicom)
    else:
        injdt = rpis[0].get('RadiopharmaceuticalStartTime')
        if injdt is not None:
            ad = ds.get('AcquisitionDate')
            if ad is not None:
                inj_datetime = dt.datetime.strptime(ad + _cleanT(injdt), StrDTdicom)
            else:
                inj_datetime = None

    if inj_datetime is None:
        return None

    acq_date_str = ds.get("SeriesDate")
    acq_time_str = _cleanT(ds.get("SeriesTime"))

    acq_datetime = dt.datetime.strptime(acq_date_str + acq_time_str, StrDTdicom)

    half_life = _float_or_none(rpis[0].get('RadionuclideHalfLife'))
    delta_t = acq_datetime - inj_datetime
    DKFactor = np.exp(-LN2 * delta_t.total_seconds() / half_life)

    inj_activity = _float_or_none(rpis[0].get('RadionuclideTotalDose'))
    if inj_activity is None:
        return None

    weight = _float_or_none(ds.get('PatientWeight'))
    if weight is None:
        return None

    return weight * 1000. / inj_activity / DKFactor


def write_resampled_vol(outdir, res_vol, res_z, rev, ds0, suv_dmax):
    """
    Write the resampled volume using the prototypical dataseet.
    Replace:
        - DICOM Tags for the series that are unique to this resampled series
        - DICOM Tags that are unique for each image
            - ImagePositionPatient with new z data
            - InstanceNumber
            - SliceLocation
            - ModificationDate, ModificationTime, see above
        - pixel_array with image derived from each slice (float) as int16 array
        - new RescaleIntercept and RescaleSlope
        -
    Parameters
    ----------
    outdir : str
        name of output directory.
    res_vol : ndarry
        data - 3D floats.
    res_z : ndarry
        resampled z-axis
    rev : bool
        true if, with increasing res_z (always the case) slices are ordered in reverse
    ds0 : Dataset
        inherited dicom object

    Returns
    -------

    """

    ds0.remove_private_tags()       # object was inherited from input

    #Start with global mods:

    suv_scale(ds0)

    ds0.DecayFactor = 1.0
    ds0.SliceSensitivityFactor = 1

    t = datetime.datetime.now()
    t_date = t.strftime("%Y%m%d")
    t_time = t.strftime("%H%M%S.%f")[:-3]

    #ds0.SeriesDate = t_date        # This can't change - time-stamps the series for acquisition time reference
    #ds0.SeriesTime = t_time

    ds0.ImageType = IMAGE_TYPE
    ds0.SeriesInstanceUID = series_instance_uid
    sd = str(ds0.get('SeriesDescription', ''))
    ds0.SeriesDescription = SERIES_DESCRIPTION_PREFIX + sd
    ds0.SeriesNumber = ds0.get('SeriesNumber', 0) + SERIES_NUMBER_BIAS

    ds0.SliceThickness = abs(res_z[1] - res_z[0])

    x0, y0 = ds0.ImagePositionPatient[0:2]

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    mxv, mnv = np.max(res_vol), np.min(res_vol)
    if mxv == mnv:
        mxv = mnv + 0.001 * abs(mnv)

    S =  (MAX16 - MIN16) / (mxv - mnv)
    Q = (MIN16 * mxv - MAX16 * mnv) / (mxv - mnv)   # or MIN16 - S * mnv

    vpx_max = (Q + S * mxv)
    vpx_min = (Q + S * mnv)

    """5/12/23: Spent some time trying different optins while watching behavior on ACR NILReader (NWH). That reader
    has a mind of it's own - changes the SUV range per slice no matter what it's set to originally.  I also can't figure
    out how Siemens is deciding to set these values - the window/center are customized to each slice but appear to
    lop off the top end. They always result in 0 at the bottom though.  GE omits these two tags on PET images so it must
    be a valid thing to do.
    In the future - it may be a nice feature to have WindowCenter and WindowWidth set so that SUV scale is [0,4].
    """

    ds0.RescaleSlope = 1. / S
    ds0.RescaleIntercept = (MAX16 * mnv - MIN16 * mxv) / (MAX16 - MIN16)  # -Q / S

    suvscl = suv_scale(ds0)
    if suvscl is None:
        if "WindowCenter" in ds0:
            del ds0.WindowCenter
        if "WindowWidth" in ds0:
            del ds0.WindowWidth
    else:
        vox_min = -ds0.RescaleIntercept / ds0.RescaleSlope
        vox_max = (suv_dmax / suvscl - ds0.RescaleIntercept) / ds0.RescaleSlope
        print('vox min, max', vox_min, vox_max)
        ds0.WindowWidth = int(vox_max - vox_min)
        ds0.WindowCenter = ds0.WindowWidth // 2

        test_val = suvscl * (ds0.RescaleIntercept + ds0.RescaleSlope * vox_max)
        print('test_val: ', test_val)

    #ds0.WindowCenter = int((vpx_min+vpx_max)/2)
    #ds0.WindowWidth = int(vpx_max - vpx_min)

    ds0.BitsAllocated = 16
    ds0.BitsStored = 16
    ds0.HighBit = 15
    ds0.PixelRepresentation = 1  # 2's complement

    t = datetime.datetime.now()
    t_date = t.strftime("%Y%m%d")
    t_time = t.strftime("%H%M%S.%f")[:-3]

    ds0.InstanceCreationDate = t_date
    ds0.InstanceCreationTime = t_time

    snum = 0
    for v, z in zip(res_vol, res_z):

        fn = os.path.join(outdir, '%4.4d' % snum)
        snum += 1

        pxa = (Q + S * v).astype('int16')

        ds0.SmallestImagePixelValue = np.min(pxa).astype('int16')
        ds0.LargestImagePixelValue = np.max(pxa).astype('int16')

        ds0.PixelData = pxa

        #print(np.max(pxa), ds0.RescaleSlope, ds0.RescaleIntercept)

        ds0.ImagePositionPatient = [x0, y0, z]
        ds0.SliceLocation = z
        if rev:
            ds0.InstanceNumber = len(res_z) - snum
            ds0.ImageIndex = len(res_z) - snum
        else:
            ds0.InstanceNumber = snum +1
            ds0.ImageIndex = snum +1
        ds0.SOPInstanceUID = generate_uid()

        ds0.save_as(fn, write_like_original=False)

def recombine(ncomb, vol, z):
    """
    perform successive recombination by averaging of each n slices.

    Parameters
    ----------
    ncomb : int
        number of slices to recombine into one
    vol : ndarray
    z : ndarray

    Returns
    -------

    res_vol : ndarray
        3D resampled data
    res_z

    """
    nres = (vol.shape[0] - 1) // ncomb + 1

    res_z0 = np.average(z[:ncomb])
    res_dz = np.average(z[ncomb:(2*ncomb)]) - res_z0

    res_vol = np.zeros((nres, vol.shape[1], vol.shape[2]))
    res_z = np.zeros(nres)
    for sl in range(nres):
        s0 = sl * ncomb
        s1 = min((sl+1) * ncomb, vol.shape[0])
        res_vol[sl] = np.average(vol[s0:s1], axis=0)
        res_z[sl] = res_z0 + sl * res_dz

    return res_vol, res_z

def reslice(vol, z, method):
    """
    Reslice in z using one of the interpolation methods:
    zoom
        - use scipy zoom function to shrink the volume.  zoom insists that fist and last samples
          remain in place, the rest are chopped up to something close to the requested zoom factor

    Parameters
    ----------
    vol : ndarray
        3D pet volume ready to be resliced
    z : ndarray
        1D z coordinates

    Returns
    -------

    res_vol : ndarray
    res_z : ndarray

    """

    zps = abs(z[1] - z[0])
    if method == 'zoom':
        z_zoom = zps / OUTPUT_SLICE_SPACING
        res_vol = snd.zoom(vol, [z_zoom, 1., 1.], grid_mode=False)
        res_z = snd.zoom(z, z_zoom, grid_mode=False)
    elif method == 'recombine':
        ncomb = max(int(OUTPUT_SLICE_SPACING / zps), 1)
        if ncomb > 1:
            res_vol, res_z = recombine(ncomb, vol, z)
        else:
            res_vol, res_z = vol, z
    elif method == 'ics':
        cs = np.cumsum(vol, axis=0)
        f = si.interp1d(z, cs, axis=0)
        dz = z[1] - z[0]
        n_new = int(dz * (len(z) + 1) / OUTPUT_SLICE_SPACING)
        res_z = z[0] - dz / 2. + OUTPUT_SLICE_SPACING / 2. + np.arange(n_new) * OUTPUT_SLICE_SPACING
        res_cs = f(res_z)
        res_vol = np.diff(res_cs, axis=0, prepend=0.) * dz  / OUTPUT_SLICE_SPACING

    else:
        raise SystemError('Something went wrong')

    return res_vol, res_z

def do_resample(indir, outdir, method, suv_dmax):

    filelist = get_input_filelist(indir)
    vol_ds = []
    for f in filelist:
        try:
            ds = dicom.read_file(f)
        except dicom.errors.InvalidDicomError:
            print('Skipping non-dicom file: %s' %f)
            continue
        vol_ds += [ds]

    print(len(vol_ds))

    vol_ds = sorted(vol_ds, key= lambda ipp: ipp.ImagePositionPatient[2]) # Sort by z-location
    # so z is monotonic, increasing

    z, rev = vol_qc(vol_ds)
    #print(z)

    scaled_vol = np.array([ds.RescaleIntercept + ds.pixel_array.astype(float) * ds.RescaleSlope for ds in vol_ds ])
    print(scaled_vol.shape)

    resampled_vol, resampled_z = reslice(scaled_vol, z, method)

    #print(resampled_z)

    print(resampled_vol.shape)

    write_resampled_vol(outdir, resampled_vol, resampled_z, rev, vol_ds[0], suv_dmax)


def _psetup():
    parser = argparse.ArgumentParser(description='Resample PET volume',
                                     epilog='Use %(prog)s {command} -h to get help on individual commands')
    parser.add_argument('input', metavar = 'D-IN', type=str, help='dicomdir or directory containing input series')
    parser.add_argument('output', metavar = 'D-OUT', type=str, help='directory to store output series')
    parser.add_argument('-m', '--method', choices=['zoom', 'recombine', 'ics'], default='ics')
    parser.add_argument('-s', '--suvmax', metavar='SUV', default=SUV_MAX_DISP, type=float, help='SUV scale maximum value')
    parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)

    return parser


if __name__ == "__main__":
    import argparse

    args = _psetup().parse_args()

    do_resample(args.input, args.output, args.method, args.suvmax)

    exit()
