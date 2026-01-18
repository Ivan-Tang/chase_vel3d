"""
Video generation module for aligned image sequences.

Generates high-quality MP4 videos from CHASE/RSM image data.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import astropy.units as u
import sunpy.map
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from matplotlib.patches import ConnectionPatch
import shutil
from concurrent.futures import ThreadPoolExecutor
from threading import Lock


# Matplotlib 在多线程下不是线程安全的，使用全局锁保护绘图和保存
_MPL_PLOT_LOCK = Lock()


def create_fullmap_video(aligned_data, rsms, 
                        output_dir='~/Solarphysics/solarphy/3d_velocity/frames/aligned_video/',
                        fps=5, figsize=(16, 12)):
    """
    Generate full-frame video from aligned image sequence.
    
    Creates MP4 video showing Ha Core and Ha Wing side-by-side
    for all frames in the aligned sequence.
    
    Parameters
    ----------
    aligned_data : list
        List of aligned spectral data cubes
    rsms : list
        List of original FITS objects (for headers)
    output_dir : str
        Output directory path
    fps : int, default=5
        Video frame rate
    figsize : tuple, default=(16, 12)
        Figure size (width, height)
    
    Returns
    -------
    output_path : str
        Path to generated MP4 file
    """
    os.makedirs(os.path.expanduser(output_dir), exist_ok=True)
    frames_dir = os.path.join(os.path.expanduser(output_dir), 'frames_tmp')
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"生成对齐全过程视频（{len(aligned_data)} 帧）...")

    ref_crpix1 = rsms[0][1].header['CRPIX1']
    ref_crpix2 = rsms[0][1].header['CRPIX2']

    def _render_full_frame(item):
        idx, (data, rsm) = item
        hacore = data[68, :, :]
        hawing = data[-10, :, :]

        obstime = rsm[1].header['DATE_OBS']
        coord_HIS = SkyCoord(0 * u.arcsec, 0 * u.arcsec, obstime=obstime,
                            observer='earth', frame=frames.Helioprojective)

        header_core = sunpy.map.make_fitswcs_header(
            hacore, coord_HIS,
            reference_pixel=[ref_crpix1, ref_crpix2] * u.pixel,
            scale=[0.5218 * 2, 0.5218 * 2] * u.arcsec / u.pixel,
            telescope='CHASE', instrument='RSM'
        )
        header_wing = sunpy.map.make_fitswcs_header(
            hawing, coord_HIS,
            reference_pixel=[ref_crpix1, ref_crpix2] * u.pixel,
            scale=[0.5218 * 2, 0.5218 * 2] * u.arcsec / u.pixel,
            telescope='CHASE', instrument='RSM'
        )

        hacore_map = sunpy.map.Map(hacore, header_core)
        hawing_map = sunpy.map.Map(hawing, header_wing)

        with _MPL_PLOT_LOCK:
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(1, 2, wspace=0.3)

            ax1 = fig.add_subplot(gs[0, 0], projection=hacore_map)
            hacore_map.plot(axes=ax1, title='Ha Core (Aligned)', cmap='afmhot',
                           vmin=0, vmax=4*hacore.mean())

            ax2 = fig.add_subplot(gs[0, 1], projection=hawing_map)
            hawing_map.plot(axes=ax2, title='Ha Wing (Aligned)', cmap='afmhot',
                           vmin=0, vmax=4*hawing.mean())

            fig.suptitle(f'Frame {idx}: {obstime}', fontsize=14, y=0.98)

            frame_path = os.path.join(frames_dir, f'frame_{idx:04d}.png')
            plt.savefig(frame_path, dpi=80, bbox_inches='tight')
            plt.close()
        return idx

    with ThreadPoolExecutor() as executor:
        for count, _ in enumerate(executor.map(_render_full_frame, enumerate(zip(aligned_data, rsms))), 1):
            if count % 5 == 0:
                print(f"  已生成 {count}/{len(aligned_data)} 帧")
    
    # Combine frames into video
    video_frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    if not video_frames:
        print("ERROR: 没有找到帧文件!")
        return None
    
    first_frame = cv2.imread(os.path.join(frames_dir, video_frames[0]))
    height, width = first_frame.shape[:2]
    
    output_path = os.path.join(os.path.expanduser(output_dir), 'aligned_full_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_name in video_frames:
        frame_path = os.path.join(frames_dir, frame_name)
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    out.release()
    print(f"✓ 全过程视频已保存: {output_path}")
    
    # Clean up temporary files
    shutil.rmtree(frames_dir)
    
    return output_path


def create_subplot_video(aligned_data, rsms, left, right, bottom, top,
                                output_dir='~/Solarphysics/solarphy/3d_velocity/frames/aligned_subplot/',
                                fps=5, figsize=(20, 10)):
    """
    Generate video with subregion detail view.
    
    Creates video showing full frame with highlighted subregion and detail view.
    
    Parameters
    ----------
    aligned_data : list
        List of aligned spectral data cubes
    rsms : list
        List of original FITS objects
    left, right, bottom, top : float
        Subregion boundaries (arcsec)
    output_dir : str
        Output directory
    fps : int, default=5
        Frame rate
    figsize : tuple, default=(20, 10)
        Figure size
    
    Returns
    -------
    output_path : str
        Path to generated MP4 file
    """
    os.makedirs(os.path.expanduser(output_dir), exist_ok=True)
    frames_dir = os.path.join(os.path.expanduser(output_dir), 'frames_tmp')
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"生成对齐子图视频（{len(aligned_data)} 帧）...")

    ang_res = 0.5218 * 2
    ref_crpix1 = rsms[0][1].header['CRPIX1']
    ref_crpix2 = rsms[0][1].header['CRPIX2']

    def _render_subplot_frame(item):
        idx, (data, rsm) = item

        obstime = rsm[1].header['DATE_OBS']

        # 支持传入已裁剪的 2D 子图（如 align_submaps_by_crpix 返回值），或原始 3D 光谱立方
        if data.ndim == 3:
            hacore = data[68, :, :]
            header_refpix = [ref_crpix1, ref_crpix2]
            coord_ref = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=frames.Helioprojective,
                                 obstime=obstime, observer='earth')
        elif data.ndim == 2:
            hacore = data
            # 子图已截取：把参考坐标设为子图中心，并将参考像素设在子图中心，避免坐标落在图外导致空白
            center_x = (left + right) / 2
            center_y = (bottom + top) / 2
            header_refpix = [hacore.shape[1] / 2, hacore.shape[0] / 2]
            coord_ref = SkyCoord(center_x * u.arcsec, center_y * u.arcsec, frame=frames.Helioprojective,
                                 obstime=obstime, observer='earth')
        else:
            raise ValueError(f"Unsupported data dimensions: {data.ndim}")

        header = sunpy.map.make_fitswcs_header(
            hacore, coord_ref,
            reference_pixel=header_refpix * u.pixel,
            scale=[ang_res, ang_res] * u.arcsec / u.pixel,
            telescope='CHASE', instrument='RSM'
        )
        hacore_map = sunpy.map.Map(hacore, header)

        # Create subregion box
        left_corner = SkyCoord(Tx=left * u.arcsec, Ty=bottom * u.arcsec,
                              frame=hacore_map.coordinate_frame)
        right_corner = SkyCoord(Tx=right * u.arcsec, Ty=top * u.arcsec,
                               frame=hacore_map.coordinate_frame)
        hacore_submap = hacore_map.submap(left_corner, top_right=right_corner).data

        # Calculate axis coordinates
        len_arc = (right - left) / 100 + 1
        tick_pixelx = list(np.linspace(0, (right - left) / ang_res, int(len_arc)))
        tick_pixely = list(np.linspace(0, (top - bottom) / ang_res, int(len_arc)))
        tick_arcsecx = [int(i) for i in np.linspace(left, right, int(len_arc))]
        tick_arcsecy = [int(i) for i in np.linspace(bottom, top, int(len_arc))]

        with _MPL_PLOT_LOCK:
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.3)

            # Main frame
            ax1 = fig.add_subplot(gs[0, 0], projection=hacore_map)
            hacore_map.plot(axes=ax1, title='', cmap='afmhot', vmin=0, vmax=4*hacore.mean())

            for coord in ax1.coords:
                coord.frame.set_linewidth(0)
                coord.set_ticks_visible(False)
                coord.set_ticklabel_visible(False)

            hacore_map.draw_quadrangle(left_corner, top_right=right_corner,
                                       edgecolor='black', lw=2)

            # Subregion
            ax2 = fig.add_subplot(gs[0, 1], projection=hacore_map)
            im = ax2.imshow(hacore_submap, origin='lower', cmap='afmhot',
                           vmin=0, vmax=4*hacore.mean())
            ax2.set_xlabel('Solar X (arcsec)', fontsize=12)
            ax2.set_ylabel('Solar Y (arcsec)', fontsize=12)
            ax2.set_xticks(tick_pixelx)
            ax2.set_xticklabels(tick_arcsecx, fontsize=10)
            ax2.set_yticks(tick_pixely)
            ax2.set_yticklabels(tick_arcsecy, fontsize=10)

            cbar = fig.colorbar(im, ax=ax2, pad=0.02)
            cbar.set_label('Count Rate', fontsize=11)

            # Connection lines
            xpix, ypix = hacore_map.world_to_pixel(right_corner)
            con1 = ConnectionPatch(
                (0, 1), (xpix.value, ypix.value), 'axes fraction', 'data',
                axesA=ax2, axesB=ax1, arrowstyle='-', color='black', lw=1.5
            )
            xpix, ypix = hacore_map.world_to_pixel(
                SkyCoord(right_corner.Tx, left_corner.Ty, frame=hacore_map.coordinate_frame)
            )
            con2 = ConnectionPatch(
                (0, 0), (xpix.value, ypix.value), 'axes fraction', 'data',
                axesA=ax2, axesB=ax1, arrowstyle='-', color='black', lw=1.5
            )
            ax2.add_artist(con1)
            ax2.add_artist(con2)

            fig.suptitle(f'Frame {idx}: {obstime} (Aligned)', fontsize=14, y=0.98)

            frame_path = os.path.join(frames_dir, f'frame_{idx:04d}.png')
            plt.savefig(frame_path, dpi=80, bbox_inches='tight')
            plt.close()
        return idx

    with ThreadPoolExecutor() as executor:
        for count, _ in enumerate(executor.map(_render_subplot_frame, enumerate(zip(aligned_data, rsms))), 1):
            if count % 5 == 0:
                print(f"  已生成 {count}/{len(aligned_data)} 帧")
    
    # Combine frames
    video_frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    if not video_frames:
        print("ERROR: 没有找到帧文件!")
        return None
    
    first_frame = cv2.imread(os.path.join(frames_dir, video_frames[0]))
    height, width = first_frame.shape[:2]
    
    output_path = os.path.join(os.path.expanduser(output_dir), 'aligned_subplot_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_name in video_frames:
        frame_path = os.path.join(frames_dir, frame_name)
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    out.release()
    print(f"✓ 子图视频已保存: {output_path}")
    
    shutil.rmtree(frames_dir)
    return output_path


def create_comparison_video(rsms, aligned_data,
                           output_dir='~/Solarphysics/solarphy/3d_velocity/frames/comparison/',
                           fps=5, figsize=(20, 10)):
    """
    Generate before/after alignment comparison video.
    
    Parameters
    ----------
    rsms : list
        Original FITS objects
    aligned_data : list
        Aligned spectral data
    output_dir : str
        Output directory
    fps : int, default=5
        Frame rate
    figsize : tuple
        Figure size
    
    Returns
    -------
    output_path : str
        Path to MP4 file
    """
    os.makedirs(os.path.expanduser(output_dir), exist_ok=True)
    frames_dir = os.path.join(os.path.expanduser(output_dir), 'frames_tmp')
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"生成对齐前后对比视频（{min(len(rsms), len(aligned_data))} 帧）...")

    ref_crpix1 = rsms[0][1].header['CRPIX1']
    ref_crpix2 = rsms[0][1].header['CRPIX2']
    total_frames = min(len(rsms), len(aligned_data))

    def _render_comparison_frame(idx):
        rsm = rsms[idx]
        aligned = aligned_data[idx]

        hacore_orig = rsm[1].data[68, :, :]
        hacore_aligned = aligned[68, :, :]

        obstime = rsm[1].header['DATE_OBS']
        coord_HIS = SkyCoord(0 * u.arcsec, 0 * u.arcsec, obstime=obstime,
                            observer='earth', frame=frames.Helioprojective)

        # Original coordinates
        header_orig = sunpy.map.make_fitswcs_header(
            hacore_orig, coord_HIS,
            reference_pixel=[rsm[1].header['CRPIX1'], rsm[1].header['CRPIX2']] * u.pixel,
            scale=[0.5218 * 2, 0.5218 * 2] * u.arcsec / u.pixel,
            telescope='CHASE', instrument='RSM'
        )

        # Aligned coordinates
        header_aligned = sunpy.map.make_fitswcs_header(
            hacore_aligned, coord_HIS,
            reference_pixel=[ref_crpix1, ref_crpix2] * u.pixel,
            scale=[0.5218 * 2, 0.5218 * 2] * u.arcsec / u.pixel,
            telescope='CHASE', instrument='RSM'
        )

        hacore_map_orig = sunpy.map.Map(hacore_orig, header_orig)
        hacore_map_aligned = sunpy.map.Map(hacore_aligned, header_aligned)

        with _MPL_PLOT_LOCK:
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(1, 2, wspace=0.3)

            ax1 = fig.add_subplot(gs[0, 0], projection=hacore_map_orig)
            hacore_map_orig.plot(axes=ax1, title='Before Alignment', cmap='afmhot',
                                vmin=0, vmax=4*hacore_orig.mean())

            ax2 = fig.add_subplot(gs[0, 1], projection=hacore_map_aligned)
            hacore_map_aligned.plot(axes=ax2, title='After Alignment', cmap='afmhot',
                                   vmin=0, vmax=4*hacore_aligned.mean())

            fig.suptitle(f'Frame {idx}: {obstime}', fontsize=14, y=0.98)

            frame_path = os.path.join(frames_dir, f'frame_{idx:04d}.png')
            plt.savefig(frame_path, dpi=80, bbox_inches='tight')
            plt.close()
        return idx

    with ThreadPoolExecutor() as executor:
        for count, _ in enumerate(executor.map(_render_comparison_frame, range(total_frames)), 1):
            if count % 5 == 0:
                print(f"  已生成 {count}/{total_frames} 帧")
    
    # Combine frames
    video_frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    if not video_frames:
        print("ERROR: 没有找到帧文件!")
        return None
    
    first_frame = cv2.imread(os.path.join(frames_dir, video_frames[0]))
    height, width = first_frame.shape[:2]
    
    output_path = os.path.join(os.path.expanduser(output_dir), 'comparison_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_name in video_frames:
        frame_path = os.path.join(frames_dir, frame_name)
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    out.release()
    print(f"✓ 对比视频已保存: {output_path}")
    
    shutil.rmtree(frames_dir)
    return output_path
