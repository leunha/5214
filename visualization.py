import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse
from tqdm import tqdm
import os
import glob

def display_paired_images(t1_array, t2_array):
    # Create figure and subplots with space for the sliders
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(bottom=0.2)  # Make room for sliders
    
    # Display initial frames (frame 0)
    im1 = ax1.imshow(t1_array[..., 0], cmap='gray')
    plt.colorbar(im1, ax=ax1)
    ax1.axis('off')
    ax1.set_title('T1')
    
    im2 = ax2.imshow(t2_array[..., 0], cmap='gray')
    plt.colorbar(im2, ax=ax2)
    ax2.axis('off')
    ax2.set_title('T2')
    
    # Add sliders
    slider_ax1 = plt.axes([0.1, 0.05, 0.3, 0.03])  # [left, bottom, width, height]
    slider_ax2 = plt.axes([0.6, 0.05, 0.3, 0.03])
    
    frame_slider1 = Slider(
        ax=slider_ax1,
        label='T1 Frame',
        valmin=0,
        valmax=t1_array.shape[-1] - 1,
        valinit=0,
        valstep=1
    )
    
    frame_slider2 = Slider(
        ax=slider_ax2,
        label='T2 Frame',
        valmin=0,
        valmax=t2_array.shape[-1] - 1,
        valinit=0,
        valstep=1
    )
    
    
    # Update functions for sliders
    def update1(val):
        frame = int(frame_slider1.val)
        im1.set_array(t1_array[..., frame])
        fig.canvas.draw_idle()
    
    def update2(val):
        frame = int(frame_slider2.val)
        im2.set_array(t2_array[..., frame])
        fig.canvas.draw_idle()
    
    frame_slider1.on_changed(update1)
    frame_slider2.on_changed(update2)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display paired T1/T2 images')
    parser.add_argument('-f', '--filename', type=str, help='Image filename (without path)')
    args = parser.parse_args()

    # Construct paths for T1 and T2 images
    t1_path = os.path.join('processed_dataset', 'IXI-T1', args.filename)
    t2_path = os.path.join('processed_dataset', 'IXI-T2', args.filename)
    
    # Load arrays
    t1_array = (np.load(t1_path)[:,:,10:-10]).transpose(2,0,1)
    t2_array = np.load(t2_path)
    
    print("T1 array shape:", t1_array.shape)
    print("T2 array shape:", t2_array.shape)
    print("T1 value range:", t1_array.min(), t1_array.max())
    print("T2 value range:", t2_array.min(), t2_array.max())
    
    display_paired_images(t1_array, t2_array)
