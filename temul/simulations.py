
from temul.signal_processing import (
    calibrate_intensity_distance_with_sublattice_roi,
    compare_two_image_and_create_filtered_image
)
from temul.model_creation import (count_atoms_in_sublattice_list,
                                  compare_count_atoms_in_sublattice_list,
                                  image_difference_intensity,
                                  create_dataframe_for_cif,
                                  image_difference_position,
                                  sort_sublattice_intensities,
                                  assign_z_height)
from temul.io import (create_dataframe_for_xyz,
                      write_cif_from_dataframe,
                      load_prismatic_mrc_with_hyperspy)

import temul.external.atomap_devel_012.api as am_dev
import pyprismatic as pr
import matplotlib.pyplot as plt
import hyperspy.api as hs
import pandas as pd
import os
from glob import glob


def simulate_and_filter_and_calibrate_with_prismatic(
        xyz_filename,
        filename,
        reference_image,
        calibration_area,
        calibration_separation,
        delta_image_filter=0.1,
        max_sigma=6,
        percent_to_nn=0.4,
        mask_radius=None,
        refine=True,
        scalebar_true=False,
        probeStep=None,
        E0=60e3,
        integrationAngleMin=0.085,
        integrationAngleMax=0.186,
        interpolationFactor=16,
        realspacePixelSize=0.0654,
        numFP=1,
        probeSemiangle=0.030,
        alphaBeamMax=0.032,
        scanWindowMin=0.0,
        scanWindowMax=1.0,
        algorithm="prism",
        numThreads=2):
    '''
    Simulate an xyz coordinate model with the PyPrismatic fast simulation
    software.

    Parameters
    ----------
    xyz_filename : string
        filename of the xyz coordinate model. Must be in the prismatic format.
        See http://prism-em.com/docs-inputs/ for more information.
    filename : string, default None
        name with which the image will be saved
    reference_image : hyperspy signal 2D
        image from which calibration information is taken, such
        as sampling, pixel height and pixel width
    calibration_area : list
        xy pixel coordinates of the image area to be used to calibrate
        the intensity. In the form [[0,0], [512,512]]
        See calibrate_intensity_distance_with_sublattice_roi()
    calibration_separation : int
        pixel separation used for the intensity calibration.
        See calibrate_intensity_distance_with_sublattice_roi()
    percent_to_nn : float, default 0.40
        Determines the boundary of the area surrounding each atomic
        column, as fraction of the distance to the nearest neighbour.
    mask_radius : float, default None
            Radius of the mask around each atom. If this is not set,
            the radius will be the distance to the nearest atom in the
            same sublattice times the `percent_to_nn` value.
            Note: if `mask_radius` is not specified, the Atom_Position objects
            must have a populated nearest_neighbor_list.
    refine, scalebar_true
        See function calibrate_intensity_distance_with_sublattice_roi()
    probeStep, E0 ... etc.
        See function simulate_with_prismatic()

    Returns
    -------
    Hyperspy Signal2D

    '''

    if len(calibration_area) != 2:
        raise ValueError('calibration_area must be two points')

    simulate_with_prismatic(xyz_filename=xyz_filename,
                            filename=filename,
                            reference_image=reference_image,
                            probeStep=probeStep,
                            E0=E0,
                            integrationAngleMin=integrationAngleMin,
                            integrationAngleMax=integrationAngleMax,
                            interpolationFactor=interpolationFactor,
                            realspacePixelSize=realspacePixelSize,
                            numFP=numFP,
                            probeSemiangle=probeSemiangle,
                            alphaBeamMax=alphaBeamMax,
                            scanWindowMin=scanWindowMin,
                            scanWindowMax=scanWindowMax,
                            algorithm=algorithm,
                            numThreads=numThreads)

    simulation = load_prismatic_mrc_with_hyperspy(
        'prism_2Doutput_' + filename + '.mrc', save_name=None)

    simulation = compare_two_image_and_create_filtered_image(
        image_to_filter=simulation,
        reference_image=reference_image,
        delta_image_filter=delta_image_filter,
        cropping_area=calibration_area,
        separation=calibration_separation,
        filename=None,
        max_sigma=max_sigma,
        percent_to_nn=percent_to_nn,
        mask_radius=mask_radius,
        refine=False)

    return(simulation)


def simulate_and_calibrate_with_prismatic(
        xyz_filename,
        filename,
        reference_image,
        calibration_area,
        calibration_separation,
        percent_to_nn=0.4,
        mask_radius=None,
        refine=True,
        scalebar_true=False,
        probeStep=None,
        E0=60e3,
        integrationAngleMin=0.085,
        integrationAngleMax=0.186,
        interpolationFactor=16,
        realspacePixelSize=0.0654,
        numFP=1,
        probeSemiangle=0.030,
        alphaBeamMax=0.032,
        scanWindowMin=0.0,
        scanWindowMax=1.0,
        algorithm="prism",
        numThreads=2):
    '''
    Simulate an xyz coordinate model with the PyPrismatic fast simulation
    software.

    Parameters
    ----------
    xyz_filename : string
        filename of the xyz coordinate model. Must be in the prismatic format.
        See http://prism-em.com/docs-inputs/ for more information.
    filename : string, default None
        name with which the image will be saved
    reference_image : hyperspy signal 2D
        image from which calibration information is taken, such
        as sampling, pixel height and pixel width
    calibration_area : list
        xy pixel coordinates of the image area to be used to calibrate
        the intensity. In the form [[0,0], [512,512]]
        See calibrate_intensity_distance_with_sublattice_roi()
    calibration_separation : int
        pixel separation used for the intensity calibration.
        See calibrate_intensity_distance_with_sublattice_roi()
    percent_to_nn : float, default 0.40
        Determines the boundary of the area surrounding each atomic
        column, as fraction of the distance to the nearest neighbour.
    mask_radius : float, default None
            Radius of the mask around each atom. If this is not set,
            the radius will be the distance to the nearest atom in the
            same sublattice times the `percent_to_nn` value.
            Note: if `mask_radius` is not specified, the Atom_Position objects
            must have a populated nearest_neighbor_list.
    refine, scalebar_true
        See function calibrate_intensity_distance_with_sublattice_roi()
    probeStep, E0 ... etc.
        See function simulate_with_prismatic()

    Returns
    -------
    Hyperspy Signal2D

    '''

    if len(calibration_area) != 2:
        raise ValueError('calibration_area must be two points')

    simulate_with_prismatic(xyz_filename=xyz_filename,
                            filename=filename,
                            reference_image=reference_image,
                            probeStep=probeStep,
                            E0=E0,
                            integrationAngleMin=integrationAngleMin,
                            integrationAngleMax=integrationAngleMax,
                            interpolationFactor=interpolationFactor,
                            realspacePixelSize=realspacePixelSize,
                            numFP=numFP,
                            probeSemiangle=probeSemiangle,
                            alphaBeamMax=alphaBeamMax,
                            scanWindowMin=scanWindowMin,
                            scanWindowMax=scanWindowMax,
                            algorithm=algorithm,
                            numThreads=numThreads)

    simulation = load_prismatic_mrc_with_hyperspy(
        'prism_2Doutput_' + filename + '.mrc', save_name=None)

    calibrate_intensity_distance_with_sublattice_roi(
        image=simulation,
        cropping_area=calibration_area,
        separation=calibration_separation,
        filename=filename,
        reference_image=reference_image,
        percent_to_nn=percent_to_nn,
        mask_radius=mask_radius,
        refine=refine,
        scalebar_true=scalebar_true)

    return(simulation)


def simulate_with_prismatic(xyz_filename,
                            filename,
                            reference_image=None,
                            probeStep=1.0,
                            E0=60e3,
                            integrationAngleMin=0.085,
                            integrationAngleMax=0.186,
                            detectorAngleStep=0.001,
                            interpolationFactor=16,
                            realspacePixelSize=0.0654,
                            numFP=1,
                            cellDimXYZ=None,
                            tileXYZ=None,
                            probeSemiangle=0.030,
                            alphaBeamMax=0.032,
                            scanWindowMin=0.0,
                            scanWindowMax=1.0,
                            algorithm="prism",
                            numThreads=2):
    '''
    Simulate an xyz coordinate model with pyprismatic
    fast simulation software.

    Parameters
    ----------

    xyz_filename : string
        filename of the xyz coordinate model. Must be in the prismatic format.
        See http://prism-em.com/docs-inputs/ for more information.
    filename : string, default None
        name with which the image will be saved
    reference_image : hyperspy signal 2D
        image from which calibration information is taken, such
        as sampling, pixel height and pixel width
    probeStep : float, default 1.0
        Should be the sampling of the image, where
        sampling = length (in angstrom)/pixels
        If you want the probeStep to be calculated from the reference image,
        set this to None.
    E0, numThreads etc.,  : Prismatic parameters
        See http://prism-em.com/docs-params/
    cellDimXYZ : tuple, default None
        A tuple of length 3. Example (2.3, 4.5, 5.6).
        If this is set to None, the cell dimension values from the .xyz file
        will be used (default). If it is specified, it will overwrite the .xyz
        file values.
    tileXYZ : tuple, default None
        A tuple of length 3. Example (5, 5, 2) would multiply the model in x
        and y by 5, and z by 2.
        Default of None is just set to (1, 1, 1)

    Returns
    -------
    Simulated image as a 2D .mrc file

    Examples
    --------
    >>> from temul.simulations import simulate_with_prismatic
    >>> import temul.example_data as example_data
    >>> file_path = path_to_example_data_MoS2_hex_prismatic()
    >>> simulate_with_prismatic(
    ...     xyz_filename=file_path,
    ...     filename='prismatic_simulation',
    ...     probeStep=1.0, reference_image=None, E0=60e3,
    ...     integrationAngleMin=0.085,
    ...     integrationAngleMax=0.186,
    ...     interpolationFactor=16,
    ...     realspacePixelSize=0.0654,
    ...     numFP=1, probeSemiangle=0.030, alphaBeamMax=0.032,
    ...     scanWindowMin=0.0, scanWindowMax=1.0,
    ...     algorithm="prism", numThreads=2)

    '''

    if '.xyz' not in xyz_filename:
        simulation_filename = xyz_filename + '.XYZ'
    else:
        simulation_filename = xyz_filename

    file_exists = os.path.isfile(simulation_filename)
    if file_exists:
        pass
    else:
        raise OSError('XYZ file not found in directory, stopping refinement')

    # param inputs, feel free to add more!!
    pr_sim = pr.Metadata(filenameAtoms=simulation_filename)

    # use the reference image to get the probe step if given
    # fix these
    if reference_image is None and probeStep is None:
        raise ValueError("Both reference_image and probeStep are None.\
            Either choose a reference image, from which a probe step can\
            be calculated, or choose a probeStep.")
    elif reference_image is not None and probeStep is not None:
        print("Note: Both reference_image and probeStep have been specified. "
              "reference_image will be used.")

    if reference_image is not None:
        real_sampling = reference_image.axes_manager[0].scale
        if reference_image.axes_manager[-1].units == 'nm':
            real_sampling_exp_angs = real_sampling * 10
        else:
            real_sampling_exp_angs = real_sampling
        if str(real_sampling_exp_angs)[-1] == '5':
            real_sampling_sim_angs = real_sampling_exp_angs + 0.000005

            pr_sim.probeStepX = pr_sim.probeStepY = round(
                real_sampling_sim_angs, 6)
        else:
            real_sampling_sim_angs = real_sampling_exp_angs + 0.000005
            pr_sim.probeStepX = pr_sim.probeStepY = round(
                real_sampling_sim_angs, 6)
    else:
        pr_sim.probeStepX = pr_sim.probeStepY = probeStep

    # if you specify cellDimXYZ, you overwrite the values from the xyz file
    if cellDimXYZ is not None:
        pr_sim.cellDimX, pr_sim.cellDimX, pr_sim.cellDimX = cellDimXYZ
    if tileXYZ is not None:
        pr_sim.tileX, pr_sim.tileY, pr_sim.tileZ = tileXYZ

    #    pr_sim.probeStepX = pr_sim.cellDimX/atom_lattice_data.shape[1]
    #    pr_sim.probeStepY = pr_sim.cellDimY/atom_lattice_data.shape[0]
    pr_sim.detectorAngleStep = detectorAngleStep
    pr_sim.save2DOutput = True
    pr_sim.save3DOutput = False

    pr_sim.E0 = E0
    pr_sim.integrationAngleMin = integrationAngleMin
    pr_sim.integrationAngleMax = integrationAngleMax
    pr_sim.interpolationFactorX = pr_sim.interpolationFactorY = \
        interpolationFactor
    pr_sim.realspacePixelSizeX = pr_sim.realspacePixelSizeY = \
        realspacePixelSize
    pr_sim.numFP = numFP
    pr_sim.probeSemiangle = probeSemiangle
    pr_sim.alphaBeamMax = alphaBeamMax  # in rads
    pr_sim.scanWindowXMin = pr_sim.scanWindowYMin = scanWindowMin
    pr_sim.scanWindowYMax = pr_sim.scanWindowXMax = scanWindowMax
    pr_sim.algorithm = algorithm
    pr_sim.numThreads = numThreads
    pr_sim.filenameOutput = filename + '.mrc'
    pr_sim.go()


# Purpose built for an in-house use-case
def image_refine_via_intensity_loop(atom_lattice,
                                    change_sublattice,
                                    calibration_separation,
                                    calibration_area,
                                    percent_to_nn,
                                    mask_radius,
                                    element_list,
                                    image_sampling,
                                    iterations,
                                    delta_image_filter,
                                    image_size_x_nm,
                                    image_size_y_nm,
                                    image_size_z_nm,
                                    simulation_filename,
                                    filename,
                                    intensity_type,
                                    intensity_refine_name='intensity_refine_',
                                    folder_name='refinement_of_intensity'):

    for sub in atom_lattice.sublattice_list:
        if sub.name == 'sub1':
            sub1 = sub
        elif sub.name == 'sub2':
            sub2 = sub
        elif sub.name == 'sub3':
            sub3 = sub
        else:
            pass

    if len(atom_lattice.image) == 1:
        # image_pixel_x = len(atom_lattice.image.data[0, :])
        # image_pixel_y = len(atom_lattice.image.data[:, 0])
        # atom_lattice_data = atom_lattice.image.data
        atom_lattice_signal = atom_lattice.image
    elif len(atom_lattice.image) > 1:
        # image_pixel_x = len(atom_lattice.image[0, :])
        # image_pixel_y = len(atom_lattice.image[:, 0])
        # atom_lattice_data = atom_lattice.image
        atom_lattice_signal = atom_lattice.signal

    '''
    Image Intensity Loop
    '''

    if len(calibration_area) != 2:
        raise ValueError('calibration_area_simulation must be two points')

    df_inten_refine = pd.DataFrame(columns=element_list)

    real_sampling_exp_angs = image_sampling * 10

    if str(real_sampling_exp_angs)[-1] == '5':
        real_sampling_sim_angs = real_sampling_exp_angs + 0.000005
    else:
        pass

    for suffix in range(1, iterations):

        loading_suffix = '_' + str(suffix)

        saving_suffix = '_' + str(suffix + 1)

        if '.xyz' in simulation_filename:
            pass
        else:
            simulation_filename = simulation_filename + '.xyz'

        file_exists = os.path.isfile(simulation_filename)
        if file_exists:
            pass
        else:
            raise OSError('XYZ file not found, stopping refinement')

        file = pr.Metadata(filenameAtoms=simulation_filename, E0=60e3)

        file.integrationAngleMin = 0.085
        file.integrationAngleMax = 0.186

        file.interpolationFactorX = file.interpolationFactorY = 16
        file.realspacePixelSizeX = file.realspacePixelSizeY = 0.0654
    #    file.probeStepX = file.cellDimX/atom_lattice_data.shape[1]
    #    file.probeStepY = file.cellDimY/atom_lattice_data.shape[0]
        file.probeStepX = round(real_sampling_sim_angs, 6)
        file.probeStepY = round(real_sampling_sim_angs, 6)
        file.numFP = 1
        file.probeSemiangle = 0.030
        file.alphaBeamMax = 0.032  # in rads
        file.detectorAngleStep = 0.001
        file.scanWindowXMin = file.scanWindowYMin = 0.0
        file.scanWindowYMax = file.scanWindowXMax = 1.0
        file.algorithm = "prism"
        file.numThreads = 2
        file.save3DOutput = False

        file.filenameOutput = intensity_refine_name + loading_suffix + ".mrc"

        file.go()

        simulation = hs.load('prism_2Doutput_' + file.filenameOutput)
        simulation.axes_manager[0].name = 'extra_dimension'
        simulation = simulation.sum('extra_dimension')
        simulation.axes_manager[0].scale = image_sampling
        simulation.axes_manager[1].scale = image_sampling

        calibrate_intensity_distance_with_sublattice_roi(
            image=simulation,
            cropping_area=calibration_area,
            separation=calibration_separation,
            filename=intensity_refine_name + "Simulation" + loading_suffix,
            percent_to_nn=percent_to_nn,
            mask_radius=mask_radius,
            scalebar_true=True)

        # simulation.plot()

        # Filter the image with Gaussian noise to get better match with
        # experiment
        simulation_new = compare_two_image_and_create_filtered_image(
            image_to_filter=simulation,
            reference_image=atom_lattice_signal,
            delta_image_filter=delta_image_filter,
            cropping_area=calibration_area,
            separation=calibration_separation,
            percent_to_nn=percent_to_nn,
            mask_radius=mask_radius,
            refine=False,
            filename=filename)

        simulation = simulation_new

        simulation.save(intensity_refine_name +
                        'Filt_Simulation' + loading_suffix + '.hspy')

        simulation.plot()
        plt.title('Filt_Simulation' + loading_suffix, fontsize=20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(
            fname=intensity_refine_name + 'Filt_Simulation' +
            loading_suffix + '.png',
            transparent=True, frameon=False, bbox_inches='tight',
            pad_inches=None, dpi=300, labels=False)
        plt.close()

        '''
        Need to add the intensity type to the image_difference_intensity
        algorithm!
        '''

        counter_before_refinement = count_atoms_in_sublattice_list(
            sublattice_list=atom_lattice.sublattice_list,
            filename=intensity_refine_name + 'Elements' + loading_suffix)

        if suffix == 1:
            df_inten_refine = df_inten_refine.append(
                counter_before_refinement, ignore_index=True).fillna(0)
        else:
            pass

        ''' Sub1 '''
        image_difference_intensity(sublattice=sub1,
                                   simulation_image=simulation,
                                   element_list=element_list,
                                   percent_to_nn=percent_to_nn,
                                   mask_radius=mask_radius,
                                   change_sublattice=change_sublattice,
                                   filename=filename)

        # sub1_info_refined = print_sublattice_elements(sub1)

        ''' Sub2 '''
        image_difference_intensity(sublattice=sub2,
                                   simulation_image=simulation,
                                   element_list=element_list,
                                   percent_to_nn=percent_to_nn,
                                   mask_radius=mask_radius,
                                   change_sublattice=change_sublattice,
                                   filename=filename)

        # sub2_info_refined = print_sublattice_elements(sub2)

        ''' Sub3 '''
        image_difference_intensity(sublattice=sub3,
                                   simulation_image=simulation,
                                   element_list=element_list,
                                   percent_to_nn=percent_to_nn,
                                   mask_radius=mask_radius,
                                   change_sublattice=change_sublattice,
                                   filename=filename)

        # sub3_info_refined = print_sublattice_elements(sub3)
        counter_after_refinement = count_atoms_in_sublattice_list(
            sublattice_list=atom_lattice.sublattice_list,
            filename=intensity_refine_name + 'Elements' + saving_suffix)

        df_inten_refine = df_inten_refine.append(
            counter_after_refinement, ignore_index=True).fillna(0)

        compare_sublattices = compare_count_atoms_in_sublattice_list(
            counter_list=[counter_before_refinement, counter_after_refinement])

        if compare_sublattices is True:
            print('Finished Refinement! No more changes.')
            break

        if suffix > 4:
            if df_inten_refine.diff(periods=2)[-4:].all(axis=1).all() is False:
                # if statement steps above:
                # .diff(periods=2) gets the difference between each row,
                # and the row two above it [-4:] slices this new difference
                # df to get the final four rows # .all(axis=1) checks if
                # all row elements are zero or NaN and gives back False
                # .all() check if all four of these results are False
                # Basically checking that the intensity refinement is
                # repeating every second iteration
                print('Finished Refinement! Repeating every second iteration.')
                break

        ''' Remake XYZ file for further refinement'''
        # loading_suffix is now saving_suffix

        create_dataframe_for_xyz(
            sublattice_list=atom_lattice.sublattice_list,
            element_list=element_list,
            x_distance=image_size_x_nm * 10,
            y_distance=image_size_y_nm * 10,
            z_distance=image_size_z_nm * 10,
            filename=filename + saving_suffix,
            header_comment='Something Something Something Dark Side')

        # dataframe_intensity = create_dataframe_for_xyz(
        #   sublattice_list=atom_lattice.sublattice_list,
        #   element_list=element_list,
        #   x_distance=image_size_x_nm*10,
        #   y_distance=image_size_y_nm*10,
        #   z_distance=image_size_z_nm*10,
        #   filename=intensity_refine_name + image_name + saving_suffix,
        #   header_comment='Something Something Something Dark Side')

        # when ready:
        example_df_cif = create_dataframe_for_cif(
            sublattice_list=atom_lattice.sublattice_list,
            element_list=element_list)

        write_cif_from_dataframe(
            dataframe=example_df_cif,
            filename=intensity_refine_name + filename + saving_suffix,
            chemical_name_common='MoSx-1Sex',
            cell_length_a=image_size_x_nm * 10,
            cell_length_b=image_size_y_nm * 10,
            cell_length_c=image_size_z_nm * 10)

    df_inten_refine.to_pickle(intensity_refine_name + 'df_inten_refine.pkl')
    df_inten_refine.to_csv(intensity_refine_name +
                           'df_inten_refine.csv', sep=',', index=False)

    # https://python-graph-gallery.com/124-spaghetti-plot/
    # https://stackoverflow.com/questions/8931268/using-colormaps-
    # to-set-color-of-line-in-matplotlib
    plt.figure()
    palette = plt.get_cmap('tab20')
    # plt.style.use('seaborn-darkgrid')
    # multiple line plot
    color_num = 0
    for df_column in df_inten_refine:
        #    print(df_column)
        color_num += 1
        plt.plot(df_inten_refine.index, df_inten_refine[df_column],
                 marker='', color=palette(color_num), linewidth=1, alpha=0.9,
                 label=df_column)

    plt.xlim(0, len(df_inten_refine.index) + 1)
    plt.legend(loc=5, ncol=1, fontsize=10,
               fancybox=True, frameon=True, framealpha=1)
    plt.title("Refinement of Atoms via Intensity \nAll Elements",
              loc='left', fontsize=16, fontweight=0)
    plt.xlabel("Refinement Iteration", fontsize=16, fontweight=0)
    plt.ylabel("Count of Element", fontsize=16, fontweight=0)
    plt.tight_layout()
    plt.savefig(fname=intensity_refine_name + 'inten_refine_all.png',
                transparent=True, frameon=False, bbox_inches='tight',
                pad_inches=None, dpi=300, labels=False)
    plt.close()

    atom_of_interest = 'Mo_1'
    # Highlight Plot
    text_position = (len(df_inten_refine.index) + 0.2) - 1
    plt.figure()
    # plt.style.use('seaborn-darkgrid')
    # multiple line plot
    for df_column in df_inten_refine:
        plt.plot(df_inten_refine.index, df_inten_refine[df_column],
                 marker='', color='grey', linewidth=1, alpha=0.4)

    plt.plot(df_inten_refine.index,
             df_inten_refine[atom_of_interest], marker='', color='orange',
             linewidth=4, alpha=0.7)

    plt.xlim(0, len(df_inten_refine.index) + 1)

    # Let's annotate the plot
    num = 0
    for i in df_inten_refine.values[len(df_inten_refine.index) - 2][1:]:
        num += 1
        name = list(df_inten_refine)[num]
        if name != atom_of_interest:
            plt.text(text_position, i, name,
                     horizontalalignment='left', size='small', color='grey')

    plt.text(text_position, df_inten_refine.Mo_1.tail(1), 'Moly',
             horizontalalignment='left', size='medium', color='orange')

    plt.title("Refinement of Atoms via Intensity",
              loc='left', fontsize=16, fontweight=0)
    plt.xlabel("Refinement Iteration", fontsize=16, fontweight=0)
    plt.ylabel("Count of Element", fontsize=16, fontweight=0)
    plt.tight_layout()
    plt.savefig(fname=intensity_refine_name + 'inten_refine.png',
                transparent=True, frameon=False, bbox_inches='tight',
                pad_inches=None, dpi=300, labels=False)
    plt.close()

    ''' ATOM LATTICE with simulation refinement '''

    atom_lattice_int_ref_name = 'Atom_Lattice_' + \
        intensity_type + '_refined' + saving_suffix

    atom_lattice_int_ref = am_dev.Atom_Lattice(
        image=atom_lattice_signal,
        name=atom_lattice_int_ref_name,
        sublattice_list=atom_lattice.sublattice_list)
    atom_lattice_int_ref.save(
        filename=intensity_refine_name +
        atom_lattice_int_ref_name + ".hdf5", overwrite=True)
    atom_lattice_int_ref.save(
        filename=atom_lattice_int_ref_name + "_intensity.hdf5", overwrite=True)

    atom_lattice_int_ref.get_sublattice_atom_list_on_image(markersize=2).plot()
    plt.title(atom_lattice_int_ref_name, fontsize=20)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(
        fname=intensity_refine_name + atom_lattice_int_ref_name + '.png',
        transparent=True, frameon=False, bbox_inches='tight',
        pad_inches=None, dpi=300, labels=False)
    plt.close()

    os.mkdir(folder_name)
    intensity_refine_filenames = glob('*' + intensity_refine_name + '*')
    for intensity_refine_file in intensity_refine_filenames:
        # print(position_refine_file, position_refine_name + '/' +
        # position_refine_file)
        os.rename(intensity_refine_file, folder_name +
                  '/' + intensity_refine_file)


# Purpose built for an in-house use-case
def image_refine_via_position_loop(image,
                                   sublattice_list,
                                   filename,
                                   xyz_filename,
                                   add_sublattice,
                                   pixel_threshold,
                                   num_peaks,
                                   image_size_x_nm,
                                   image_size_y_nm,
                                   image_size_z_nm,
                                   calibration_area,
                                   calibration_separation,
                                   element_list,
                                   element_list_new_sub,
                                   middle_intensity_list,
                                   limit_intensity_list,
                                   delta_image_filter=0.5,
                                   intensity_type='max',
                                   scalar_method='mode',
                                   remove_background_method=None,
                                   background_sublattice=None,
                                   num_points=3,
                                   percent_to_nn=0.4,
                                   mask_radius=None,
                                   iterations=10,
                                   max_sigma=10,
                                   E0=60e3,
                                   integrationAngleMin=0.085,
                                   integrationAngleMax=0.186,
                                   interpolationFactor=16,
                                   realspacePixelSize=0.0654,
                                   numFP=1,
                                   probeSemiangle=0.030,
                                   alphaBeamMax=0.032,
                                   scanWindowMin=0.0,
                                   scanWindowMax=1.0,
                                   algorithm="prism",
                                   numThreads=2
                                   ):
    ''' Image Position Loop '''

    df_position_refine = pd.DataFrame(columns=element_list)
    new_subs = []

    create_dataframe_for_xyz(
        sublattice_list=sublattice_list,
        element_list=element_list,
        x_distance=image_size_x_nm * 10,
        y_distance=image_size_y_nm * 10,
        z_distance=image_size_z_nm * 10,
        filename=xyz_filename + '01',
        header_comment=filename)

    for suffix in range(1, iterations):

        loading_suffix = '_' + str(suffix).zfill(2)
        saving_suffix = '_' + str(suffix + 1).zfill(2)
        simulation_filename = xyz_filename + loading_suffix + '.XYZ'

        simulation = simulate_and_calibrate_with_prismatic(
            reference_image=image,
            xyz_filename=simulation_filename,
            calibration_area=calibration_area,
            calibration_separation=calibration_separation,
            filename=filename,
            percent_to_nn=percent_to_nn,
            mask_radius=mask_radius,
            E0=E0,
            integrationAngleMin=integrationAngleMin,
            integrationAngleMax=integrationAngleMax,
            interpolationFactor=interpolationFactor,
            realspacePixelSize=realspacePixelSize,
            numFP=numFP,
            probeSemiangle=probeSemiangle,
            alphaBeamMax=alphaBeamMax,
            scanWindowMin=scanWindowMin,
            scanWindowMax=scanWindowMax,
            algorithm=algorithm,
            numThreads=numThreads)

        # Filter the image with Gaussian noise to get better match with
        # experiment
        simulation_new = compare_two_image_and_create_filtered_image(
            image_to_filter=simulation,
            reference_image=image,
            filename=filename + loading_suffix,
            delta_image_filter=delta_image_filter,
            cropping_area=calibration_area,
            separation=calibration_separation,
            percent_to_nn=percent_to_nn,
            mask_radius=mask_radius,
            max_sigma=max_sigma,
            refine=False)

        simulation = simulation_new

        simulation.save('filt_sim_' + filename + loading_suffix + '.hspy')

        # simulation.plot()
        # plt.title('Filtered_Simulation' + filename +
        #           loading_suffix, fontsize=20)
        # plt.gca().axes.get_xaxis().set_visible(False)
        # plt.gca().axes.get_yaxis().set_visible(False)
        # plt.tight_layout()
        # plt.savefig(fname='filt_sim_' + filename + loading_suffix + '.png',
        #             transparent=True, frameon=False, bbox_inches='tight',
        #             pad_inches=None, dpi=300, labels=False)
        # plt.close()

        counter_before_refinement = count_atoms_in_sublattice_list(
            sublattice_list=sublattice_list,
            filename=None)

        if suffix == 1:
            df_position_refine = df_position_refine.append(
                counter_before_refinement, ignore_index=True).fillna(0)
        else:
            pass

        sub_new = image_difference_position(
            sublattice_list=sublattice_list,
            simulation_image=simulation,
            pixel_threshold=pixel_threshold,
            filename=None,
            mask_radius=mask_radius,
            num_peaks=num_peaks,
            add_sublattice=add_sublattice,
            sublattice_name='sub_new' + loading_suffix)

        if isinstance(sub_new, sublattice_list[0]):

            new_subs.append(sub_new)
            sublattice_list += [new_subs[-1]]

            sub_new.get_atom_list_on_image(markersize=2).plot()
            plt.title(sub_new.name, fontsize=20)
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.tight_layout()
            plt.savefig(fname=sub_new.name + filename + '.png',
                        transparent=True, frameon=False, bbox_inches='tight',
                        pad_inches=None, dpi=300, labels=False)
            plt.close()

            sort_sublattice_intensities(
                sublattice=sub_new,
                intensity_type=intensity_type,
                middle_intensity_list=middle_intensity_list,
                limit_intensity_list=limit_intensity_list,
                element_list=element_list_new_sub,
                scalar_method=scalar_method,
                remove_background_method=remove_background_method,
                background_sublattice=background_sublattice,
                num_points=num_points,
                percent_to_nn=percent_to_nn,
                mask_radius=mask_radius)
            '''
            need to make mask_radius for this too, for the new sublattice
            '''

            assign_z_height(sublattice=sub_new,
                            lattice_type='chalcogen',
                            material='mos2_one_layer')

    #        sub_new_info = print_sublattice_elements(sub_new)

        elif sub_new is None:
            print('All new sublattices have been added!')
            break

        counter_after_refinement = count_atoms_in_sublattice_list(
            sublattice_list=sublattice_list,
            filename=None)

        df_position_refine = df_position_refine.append(
            counter_after_refinement, ignore_index=True).fillna(0)

        ''' Remake XYZ file for further refinement'''
        # loading_suffix is now saving_suffix

        create_dataframe_for_xyz(
            sublattice_list=sublattice_list,
            element_list=element_list,
            x_distance=image_size_x_nm * 10,
            y_distance=image_size_y_nm * 10,
            z_distance=image_size_z_nm * 10,
            filename=xyz_filename + filename + saving_suffix,
            header_comment=filename)

    df_position_refine.to_csv(filename + '.csv', sep=',', index=False)

    '''Save Atom Lattice Object'''
    atom_lattice = am_dev.Atom_Lattice(image=image.data,
                                       name='All Sublattices ' + filename,
                                       sublattice_list=sublattice_list)
    atom_lattice.save(filename="Atom_Lattice_" +
                      filename + ".hdf5", overwrite=True)

    folder_name = filename + "_pos_ref_data"
    os.mkdir(folder_name)
    position_refine_filenames = glob('*' + filename + '*')
    for position_refine_file in position_refine_filenames:
        os.rename(position_refine_file, folder_name +
                  '/' + position_refine_file)
