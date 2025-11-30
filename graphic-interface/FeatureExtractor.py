import numpy as np
from skimage.feature import local_binary_pattern, hog as sk_hog
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops, moments_hu, euler_number
from scipy.ndimage import convolve

class FeatureExtractor:
    def __init__(self):
        pass
    
    @staticmethod
    def zoning(img: np.ndarray) -> np.ndarray:
        if img is None:
            return np.zeros(72, dtype=np.float32)
        if img.ndim == 1:
            side = int(np.sqrt(img.size))
            img = img.reshape(side, side)
        img = img.astype(np.float32, copy=False)
        bin_img = (img > 0).astype(np.float32)
        h, w = bin_img.shape
        gh, gw = h // 4, w // 4
        densities = []
        for i in range(4):
            for j in range(4):
                patch = bin_img[i*gh:(i+1)*gh, j*gw:(j+1)*gw]
                densities.append(patch.mean())
        proj_h = bin_img.sum(axis=1)
        proj_v = bin_img.sum(axis=0)
        proj_h = proj_h / (proj_h.sum() + 1e-7)
        proj_v = proj_v / (proj_v.sum() + 1e-7)
        return np.concatenate([np.array(densities, dtype=np.float32),
                            proj_h.astype(np.float32),
                            proj_v.astype(np.float32)]).astype(np.float32, copy=False)

    @staticmethod
    def hog(img: np.ndarray) -> np.ndarray:
        if img is None:
            return np.zeros(1, dtype=np.float32)
        if img.ndim == 1:
            side = int(np.sqrt(img.size))
            img = img.reshape(side, side)
        img = img.astype(np.float32, copy=False)
        if img.max() > 1.0:
            img = img / 255.0
        features = sk_hog(img,
                        orientations=9,
                        pixels_per_cell=(4, 4),
                        cells_per_block=(2, 2),
                        block_norm='L2-Hys',
                        transform_sqrt=True,
                        visualize=False,
                        feature_vector=True)
        return features.astype(np.float32, copy=False)

    @staticmethod
    def hu(img: np.ndarray) -> np.ndarray:
        if img is None:
            return np.zeros(7, dtype=np.float32)
        if img.ndim == 1:
            side = int(np.sqrt(img.size))
            img = img.reshape(side, side)
        bin_img = (img > 0).astype(np.float32)
        vals = moments_hu(bin_img)
        vals = -np.sign(vals) * np.log10(np.abs(vals) + 1e-10)
        return vals.astype(np.float32, copy=False)

    @staticmethod
    def lbp(img: np.ndarray) -> np.ndarray:
        if img is None:
            return np.zeros(10, dtype=np.float32)
        if img.ndim == 1:
            side = int(np.sqrt(img.size))
            img = img.reshape(side, side)
        if img.dtype.kind == 'f':
            if img.max() <= 1.0:
                img_uint = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                img_uint = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img_uint = img.astype(np.uint8, copy=False)
        lbp_img = local_binary_pattern(img_uint, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp_img.ravel(), bins=10, range=(0, 10))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        return hist.astype(np.float32, copy=False)

    @staticmethod
    def skeleton(img: np.ndarray) -> np.ndarray:
        if img is None:
            return np.zeros(4, dtype=np.float32)
        if img.ndim == 1:
            side = int(np.sqrt(img.size))
            img = img.reshape(side, side)
        bin_img = (img > 0).astype(np.uint8)
        skel = skeletonize(bin_img > 0)
        skel_uint = skel.astype(np.uint8)
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
        neighbor_count = convolve(skel_uint, kernel, mode='constant', cval=0)
        endpoints = np.sum((skel_uint == 1) & (neighbor_count == 1))
        branches = np.sum((skel_uint == 1) & (neighbor_count >= 3))
        length = skel_uint.sum()
        area = bin_img.sum()
        ratio = length / (area + 1e-7)
        return np.array([length, endpoints, branches, ratio], dtype=np.float32)

    @staticmethod
    def countours(img: np.ndarray) -> np.ndarray:
        if img is None:
            return np.zeros(6, dtype=np.float32)
        if img.ndim == 1:
            side = int(np.sqrt(img.size))
            img = img.reshape(side, side)
        bin_img = (img > 0).astype(np.uint8)
        lbl = label(bin_img)
        props = regionprops(lbl)
        if not props:
            return np.zeros(6, dtype=np.float32)
        main = max(props, key=lambda r: r.area)
        minr, minc, maxr, maxc = main.bbox
        height = maxr - minr
        width = maxc - minc
        aspect = width / (height + 1e-7)
        perimeter = main.perimeter if hasattr(main, 'perimeter') else 0.0
        feats = [main.area, main.eccentricity, main.extent, main.solidity, aspect, perimeter]
        return np.array(feats, dtype=np.float32)

    @staticmethod
    def pixel_run_length(img: np.ndarray) -> np.ndarray:
        if img is None:
            return np.zeros(6, dtype=np.float32)
        if img.ndim == 1:
            side = int(np.sqrt(img.size))
            img = img.reshape(side, side)
        bin_img = (img > 0).astype(np.uint8)

        def runs_1d(arr):
            runs = []
            count = 0
            for v in arr:
                if v == 1:
                    count += 1
                elif count > 0:
                    runs.append(count)
                    count = 0
            if count > 0:
                runs.append(count)
            if not runs:
                return [0]
            return runs

        horiz_runs = []
        for row in bin_img:
            horiz_runs.extend(runs_1d(row))
        vert_runs = []
        for col in bin_img.T:
            vert_runs.extend(runs_1d(col))

        horiz_runs = np.array(horiz_runs, dtype=np.float32)
        vert_runs = np.array(vert_runs, dtype=np.float32)

        def stats(vec):
            return np.array([vec.mean(), vec.std(), vec.max()], dtype=np.float32) if vec.size else np.zeros(3, dtype=np.float32)

        return np.concatenate([stats(horiz_runs), stats(vert_runs)]).astype(np.float32, copy=False)

    @staticmethod
    def connectivity(img: np.ndarray) -> np.ndarray:
        if img is None:
            return np.zeros(2, dtype=np.float32)
        if img.ndim == 1:
            side = int(np.sqrt(img.size))
            img = img.reshape(side, side)
        bin_img = (img > 0).astype(np.uint8)
        lbl, n_components = label(bin_img, return_num=True)
        n_components = int(n_components)
        try:
            euler = float(euler_number(bin_img))
        except Exception:
            props = regionprops(lbl)
            euler = float(sum([p.euler_number for p in props])) if props else 0.0
        return np.array([n_components, euler], dtype=np.float32)

    @staticmethod
    def geometry(img: np.ndarray) -> np.ndarray:
        if img is None:
            return np.zeros(7, dtype=np.float32)
        if img.ndim == 1:
            side = int(np.sqrt(img.size))
            img = img.reshape(side, side)
        bin_img = (img > 0).astype(np.uint8)
        lbl = label(bin_img)
        props = regionprops(lbl)
        if not props:
            return np.zeros(7, dtype=np.float32)
        main = max(props, key=lambda r: r.area)
        minr, minc, maxr, maxc = main.bbox
        cy, cx = main.centroid
        coords = np.column_stack(np.nonzero(bin_img))
        if coords.size == 0:
            inertia = 0.0
        else:
            dy = coords[:,0] - cy
            dx = coords[:,1] - cx
            inertia = float(np.sum(dx*dx + dy*dy)) / (coords.shape[0] + 1e-7)
        return np.array([minr, minc, maxr, maxc, cy, cx, inertia], dtype=np.float32)

    @staticmethod
    def projections(img: np.ndarray) -> np.ndarray:
        if img is None:
            return np.zeros(24, dtype=np.float32)
        if img.ndim == 1:
            side = int(np.sqrt(img.size))
            img = img.reshape(side, side)
        bin_img = (img > 0).astype(np.float32)
        n = bin_img.shape[0]
        proj_h = bin_img.sum(axis=1)
        proj_v = bin_img.sum(axis=0)
        h_hist, _ = np.histogram(proj_h, bins=10, range=(0, n))
        v_hist, _ = np.histogram(proj_v, bins=10, range=(0, n))
        h_hist = h_hist.astype(np.float32) / (h_hist.sum() + 1e-7)
        v_hist = v_hist.astype(np.float32) / (v_hist.sum() + 1e-7)
        diag_sums = []
        for k in range(-n+1, n):
            diag_sums.append(np.sum(np.diag(bin_img, k=k)))
        diag_sums = np.array(diag_sums, dtype=np.float32)
        m = len(diag_sums)
        group_size = m // 4
        diag_feats = []
        for i in range(4):
            start = i*group_size
            end = (i+1)*group_size if i < 3 else m
            s = diag_sums[start:end].sum()
            diag_feats.append(s)
        diag_feats = np.array(diag_feats, dtype=np.float32)
        if diag_feats.sum() > 0:
            diag_feats = diag_feats / (diag_feats.sum() + 1e-7)
        return np.concatenate([h_hist, v_hist, diag_feats]).astype(np.float32, copy=False)

    @staticmethod
    def density_quadrants(img: np.ndarray) -> np.ndarray:
        if img is None:
            return np.zeros(2, dtype=np.float32)
        if img.ndim == 1:
            side = int(np.sqrt(img.size))
            img = img.reshape(side, side)
        bin_img = (img > 0).astype(np.float32)
        h, w = bin_img.shape
        mid_h = h // 2
        mid_w = w // 2
        tl = bin_img[:mid_h, :mid_w].sum()
        tr = bin_img[:mid_h, mid_w:].sum()
        bl = bin_img[mid_h:, :mid_w].sum()
        br = bin_img[mid_h:, mid_w:].sum()
        left = (tl + bl) / (bin_img.sum() + 1e-7)
        right = (tr + br) / (bin_img.sum() + 1e-7)
        return np.array([left, right], dtype=np.float32)

    @staticmethod
    def extract_features_img(img: np.ndarray) -> np.ndarray:
        fz = FeatureExtractor.zoning(img)
        fh = FeatureExtractor.hog(img)
        fhu = FeatureExtractor.hu(img)
        fl = FeatureExtractor.lbp(img)
        fs = FeatureExtractor.skeleton(img)
        fc = FeatureExtractor.countours(img)
        fr = FeatureExtractor.pixel_run_length(img)
        fconn = FeatureExtractor.connectivity(img)
        fgeo = FeatureExtractor.geometry(img)
        fproj = FeatureExtractor.projections(img)
        fdens = FeatureExtractor.density_quadrants(img)
        feats = np.concatenate([fz, fh, fhu, fl, fs, fc, fr, fconn, fgeo, fproj, fdens])
        return feats.astype(np.float32, copy=False)