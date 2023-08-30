"""
Class implementing a PMT dataset for CNNs in h5 format
Modified from mPMT dataset for use with single PMTs
"""

# torch imports
from torch import from_numpy
from torch import flip

# generic imports
import numpy as np

# WatChMaL imports
from WatChMaL.watchmal.dataset.h5_dataset import H5Dataset
import WatChMaL.watchmal.dataset.data_utils as du
from WatChMaL.watchmal.engine.tensor_plot_maker import *

i = 1

class CNNDataset(H5Dataset):
    """
    This class loads PMT hit data from an HDF5 file and provides events formatted for CNNs, where the 3D data tensor's
    first dimension is over the channels, corresponding to hit time and/or charge, and the second and third dimensions
    are the height and width of the CNN image. Each pixel of the image corresponds to one PMT, with PMTs arrange in an
    event-display-like format.
    """

    def __init__(self, h5file, pmt_positions_file, use_times=True, use_charges=True, transforms=None, one_indexed=False):
        """
        Constructs a dataset for CNN data. Event hit data is read in from the HDF5 file and the PMT charge and/or time
        data is formatted into an event-display-like image for input to a CNN. Each pixel of the image corresponds to
        one PMT and the channels correspond to charge and/or time at each PMT. The PMTs are placed in the image
        according to a mapping provided by the numpy array in the `pmt_positions_file`.

        Parameters
        ----------
        h5file: string
            Location of the HDF5 file containing the event data
        pmt_positions_file: string
            Location of an npz file containing the mapping from PMT IDs to CNN image pixel locations
        use_times: bool
            Whether to use PMT hit times as one of the initial CNN image channels. True by default.
        use_charges: bool
            Whether to use PMT hit charges as one of the initial CNN image channels. True by default.
        transforms
            List of random transforms to apply to data before passing to CNN for data augmentation. Currently unused for
            this dataset.
        one_indexed: bool
            Whether the PMT IDs in the H5 file are indexed starting at 1 (like SK tube numbers) or 0 (like WCSim PMT
            indexes). By default, zero-indexing is assumed.
        """
        super().__init__(h5file)
        #print('np.load whatever find this ______________________', np.load(pmt_positions_file))
        self.pmt_positions = np.load(pmt_positions_file)
        self.use_times = use_times
        self.use_charges = use_charges
        self.data_size = np.max(self.pmt_positions, axis=0) + 1
        self.barrel_rows = [row for row in range(self.data_size[0]) if
                            np.count_nonzero(self.pmt_positions[:, 0] == row) == self.data_size[1]]
        
        self.transforms = du.get_transformations(self, transforms)

        self.one_indexed = one_indexed
            
        self.counter = 0
        
        n_channels = 0
        if use_times:
            n_channels += 1
        if use_charges:
            n_channels += 1
        if n_channels == 0:
            raise Exception("Please set 'use_times' and/or 'use_charges' to 'True' in your data config.")
        
            
        self.data_size = np.insert(self.data_size, 0, n_channels)

    def process_data(self, hit_pmts, hit_times, hit_charges, double_cover = None, transforms = None):
        """
        Returns event data from dataset associated with a specific index

        Parameters
        ----------
        hit_pmts: array_like of int
            Array of hit PMT IDs
        hit_times: array_like of float
            Array of PMT hit times
        hit_charges: array_like of float
            Array of PMT hit charges
        
        Returns
        -------
        data: ndarray
            Array in image-like format (channels, rows, columns) for input to CNN network.
        """
        #self.transforms = du.get_transformations(self, transforms)
        
        if self.one_indexed:
            hit_pmts = hit_pmts-1  # SK cable numbers start at 1

        hit_rows = self.pmt_positions[hit_pmts, 0]
        hit_cols = self.pmt_positions[hit_pmts, 1]

        data = np.zeros(self.data_size, dtype=np.float32)
        
        #print('self.transforms', self.transforms)

        #if double_cover is in self.transforms:
            #data = self.double_cover(data)

        if self.use_times and self.use_charges:
            data[0, hit_rows, hit_cols] = hit_times
            data[1, hit_rows, hit_cols] = hit_charges
        elif self.use_times:
            data[0, hit_rows, hit_cols] = hit_times
        else:
            data[0, hit_rows, hit_cols] = hit_charges

        return data

    def __getitem__(self, item):

        data_dict = super().__getitem__(item)
        processed_data = from_numpy(self.process_data(self.event_hit_pmts, self.event_hit_times, self.event_hit_charges))
        processed_data = du.apply_random_transformations(self.transforms, processed_data)
        processed_data = self.double_cover(processed_data)
        data_dict["data"] = processed_data
        
        self.counter = self.counter + 1
        return data_dict


    def center(self, data):
        """
        Centers the mean of the charges to the center of the picture
        """
        return image_mover(data)

    def horizontal_flip(self, data):
        """
        Takes image-like data and returns the data after applying a horizontal flip to the image.
        The channels of the PMTs within mPMTs also have the appropriate permutation applied.
        """
        #print('applying horizontal flip')
        return flip(data[:, :], [2])

    def vertical_flip(self, data):
        """
        Takes image-like data and returns the data after applying a vertical flip to the image.
        The channels of the PMTs within mPMTs also have the appropriate permutation applied.
        """
        #print('applying vertical flip')
        return flip(data[:, :], [1])

    def flip_180(self, data):
        """
        Takes image-like data and returns the data after applying both a horizontal flip to the image. This is
        equivalent to a 180-degree rotation of the image.
        The channels of the PMTs within mPMTs also have the appropriate permutation applied.
        """
        #print('applying 180 flip')
        return self.horizontal_flip(self.vertical_flip(data))
 
    def front_back_reflection(self, data):
        """
        Takes CNN input data in event-display-like format and returns the data with horizontal flip of the left and
        right halves of the barrels and vertical flip of the endcaps. This is equivalent to reflecting the detector
        swapping the front and back of the event-display view. The channels of the PMTs within mPMTs also have the
        appropriate permutation applied.
        """
        #print('front back reflected')
        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]
        radius_endcap = barrel_row_start//2                     # 5
        half_barrel_width = data.shape[2]//2                    # 20
        l_endcap_index = half_barrel_width - radius_endcap      # 15
        r_endcap_index = half_barrel_width + radius_endcap      # 25
        
        transform_data = data.clone()

        # Take out the left and right halves of the barrel
        left_barrel = data[:, self.barrel_rows, :half_barrel_width]
        right_barrel = data[:, self.barrel_rows, half_barrel_width:]
        # Horizontal flip of the left and right halves of barrel
        transform_data[:, self.barrel_rows, :half_barrel_width] = self.horizontal_flip(left_barrel)
        transform_data[:, self.barrel_rows, half_barrel_width:] = self.horizontal_flip(right_barrel)

        # Take out the top and bottom endcaps
        top_endcap = data[:, :barrel_row_start, l_endcap_index:r_endcap_index]
        bottom_endcap = data[:, barrel_row_end+1:, l_endcap_index:r_endcap_index]
        # Vertical flip of the top and bottom endcaps
        transform_data[:, :barrel_row_start, l_endcap_index:r_endcap_index] = self.vertical_flip(top_endcap)
        transform_data[:, barrel_row_end+1:, l_endcap_index:r_endcap_index] = self.vertical_flip(bottom_endcap)

        return transform_data

    def rotation180(self, data):
        """
        Takes CNN input data in event-display-like format and returns the data with horizontal and vertical flip of the
        endcaps and shifting of the barrel rows by half the width. This is equivalent to a 180-degree rotation of the
        detector about its axis. The channels of the PMTs within mPMTs also have the appropriate permutation applied.
        """
        #print('rotated 180')
        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]   # 10,18 respectively
        radius_endcap = barrel_row_start//2                 # 5
        l_endcap_index = data.shape[2]//2 - radius_endcap   # 15
        r_endcap_index = data.shape[2]//2 + radius_endcap   # 25   

        transform_data = data.clone()

        # Take out the top and bottom endcaps
        top_endcap = data[:, :barrel_row_start, l_endcap_index:r_endcap_index]
        bottom_endcap = data[:, barrel_row_end+1:, l_endcap_index:r_endcap_index]
        # Vertical and horizontal flips of the endcaps
        transform_data[:, :barrel_row_start, l_endcap_index:r_endcap_index] = self.flip_180(top_endcap)
        transform_data[:, barrel_row_end+1:, l_endcap_index:r_endcap_index] = self.flip_180(bottom_endcap)

        # Swap the left and right halves of the barrel
        transform_data[:, self.barrel_rows, :] = torch.roll(transform_data[:, self.barrel_rows, :], 20, 2)

        return transform_data
    
    def mpmt_padding(self, data):
        """
        Takes CNN input data in event-display-like format and returns the data with part of the barrel duplicated to one
        side, and copies of the end-caps duplicated, rotated 180 degrees and with PMT channels in the mPMTs permuted, to
        provide two 'views' of the detect in one image.
        """
        #print('mpmt padding')
        w = data.shape[2]
        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]
        l_endcap_index = w//2 - 5
        r_endcap_index = w//2 + 4

        padded_data = torch.cat((data, torch.zeros_like(data[:, :w//2])), dim=2)
        padded_data[:, self.barrel_rows, w:] = data[:, self.barrel_rows, :w//2]

        # Take out the top and bottom endcaps
        top_endcap = data[:, :barrel_row_start, l_endcap_index:r_endcap_index+1]
        bottom_endcap = data[:, barrel_row_end+1:, l_endcap_index:r_endcap_index+1]

        padded_data[barrel_row_start, l_endcap_index+w//2:r_endcap_index+w//2+1] = self.flip_180(top_endcap)
        padded_data[barrel_row_end+1:, l_endcap_index+w//2:r_endcap_index+w//2+1] = self.flip_180(bottom_endcap)

        return padded_data

    def double_cover(self, data):
        """
        Takes CNN input data in event-display-like format and returns the data with all parts of the detector duplicated
        and rearranged to provide a double-cover of the image, providing two 'views' of the detector from a single image
        with less blank space and physically meaningful cyclic boundary conditions at the edges of the image.

        The transformation looks something like the following, where PMTs on the end caps are numbered and PMTs on the
        barrel are letters:
        ```
                             CBALKJIHGFED
             01                01    32
             23                23    10
        ABCDEFGHIJKL   -->   DEFGHIJKLABC
        MNOPQRSTUVWX         PQRSTUVWXMNO
             45                45    76
             67                67    54
                             ONMXWVUSTRQP
        ```
        """
        #print('double cover')
        w = data.shape[2]                                                                            
        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]
        radius_endcap = barrel_row_start//2
        half_barrel_width, quarter_barrel_width = w//2, w//4

        # Step - 1 : Roll the tensor so that the first quarter is the last quarter
        padded_data = torch.roll(data, -quarter_barrel_width, 2)

        # Step - 2 : Copy the endcaps and paste 3 quarters from the start, after flipping 180 
        l1_endcap_index = half_barrel_width - radius_endcap - quarter_barrel_width
        r1_endcap_index = l1_endcap_index + 2*radius_endcap
        l2_endcap_index = l1_endcap_index+half_barrel_width
        r2_endcap_index = r1_endcap_index+half_barrel_width

        top_endcap = padded_data[:, :barrel_row_start, l1_endcap_index:r1_endcap_index]
        bottom_endcap = padded_data[:, barrel_row_end+1:, l1_endcap_index:r1_endcap_index]
        
        padded_data[:, :barrel_row_start, l2_endcap_index:r2_endcap_index] = self.flip_180(top_endcap)
        padded_data[:, barrel_row_end+1:, l2_endcap_index:r2_endcap_index] = self.flip_180(bottom_endcap)
        
        # Step - 3 : Rotate the top and bottom half of barrel and concat them to the top and bottom respectively
        barrel_rows_top, barrel_rows_bottom = np.array_split(self.barrel_rows, 2)
        barrel_top_half, barrel_bottom_half = padded_data[:, barrel_rows_top, :], padded_data[:, barrel_rows_bottom, :]
        
        concat_order = (self.flip_180(barrel_top_half), 
                        padded_data,
                        self.flip_180(barrel_bottom_half))

        padded_data = torch.cat(concat_order, dim=1)

        return padded_data
