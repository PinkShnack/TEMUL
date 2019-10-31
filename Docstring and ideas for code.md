# Docstring Example

```
'''
Explanation of function

Parameters
----------

Returns
-------

Examples
--------
'''
```

# To Dos:

1. sort unnecessary files into current correct files - done
2. make sure imports are linking each module
3. create api file that imports everything, or two
    3a. One for everything - done
    3b. One for everything except prismatic -done
4. verify api files import what you want
    4a. just import the no_prismatic api into api with *
5. make checklist for pull requests.
    5a. should you add the function to the api/api_no_prismatic?
    5b. docstring with examples?
    5c. works in site-packages?
6. add fft masking code
    6a. PR on atomap for log plotting of add_atoms_with_gui()    



# Ideas for code development


Absolutely Need to Develop:

*Update all to newest version of atomap and hyperspy.

gaussian fitting function of multiple histogram regions, with ouput/log: 
    for the fitting of gaussians to 1D intensity plot/histogram;
    each time it fails or suceeds, print out the response. and finally 
    print out a log of the whole fitting procedure.

intensity refine function:
    make cleaner, layout as below, maybe split up if needed/better.

Errors and uncertainties
    define the gaussian fitting error and uncertainty
    define the intensity error and uncertainty
    log the error between images (exp vs. sim)

lorentzian filter for simulation, increase time and frozen Phonons also

max, mean, min for calibrate and other functions that don't have it already.
    

For the image_size_x,y,z have the input as a list of len 3 (x,y,z)
Same for the create cif and xyz data

Similar for the mask radii for intensity refine. Have it on loop, going through
a sublattice for each mask radius. Could have a check to see if they are the 
same length. If not, tell the user that the final mask_radius will be used for
every other sublattice after the last mask radius entered.
In other words: if we have 5 sublattices, and input a list of 3 mask_radius,
we simply use the final mask radius for the last 3 sublattices in the loop! 

For the refine loops, each loop, print out the time taken in minutes. Then at
the end, print out the entire time taken.

Make it easy to have the mpl plots saved or not. Just save the data should be 
an option. It's easy to plot the mpl or hs plots after in a loop!

Position Refine;
get prismatic sampling edit
open a pandas df for tracking changes
create xyz file for sublattices given

create loop
    set up simulation
    do simulation
    load simulation
    calibrate simulation

    all of the above done in the simulate_with_prismatic function

    filter simulation
    save calibrated and filtered simulation

    

counter before refinement
image diff position to create new sublattice
    plot new sub
    assign elements and Z height
    save new sub

counter after refinement
compare sublattice
    dont think I need to compare counters for image diff position? 
    IT counts the number of elements before a new sublattice is added. It isn't 
    going to the same if a new sublattice is added, and it is going to return 
    sub_new=None anyway and break...

    logging the counts before the first and after each iteration is good 
    though, because we can save the evolution of the refinement in a pandas 
    csv file.


create new xyz file for next iteration
save counts as a csv file
create a new folder in which to place all of this refinement.
    make sure the filename isn't overlapping with another tag in your folder,
    otherwise those files would also be moved.
move the xyz files in 


Add functions to polarisation.py for domain wall mapping.
> centre_colormap for the LNO domain and any diverging, see if it's useful.



# Order to upload to atomap

max min mean total with tests


