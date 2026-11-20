% Tumour-boundary shell masks for the Erasmus Glioma Database (EGD).
%
% For every patient, the supplied whole-tumour mask is dilated and eroded
% with a spherical structuring element of radius 3 voxels. The shell is the
% difference between both (it spans voxels just inside and just outside the
% segmented boundary). The result is written next to the original mask as
% PERIPHERY_MASK.nii.gz and later used by PyRadiomics.
%
% Requires the Image Processing Toolbox (strel, imdilate, imerode).
% Run from this folder; base_path points to the EGD download.

% Base path to the dataset
base_path = '../original_datasets/dataset_EGD/';

% Define the folder name pattern for patients
patient_folder_pattern = 'MR_EGD-*';

% Get a list of all patient folders
patient_folders = dir(fullfile(base_path, patient_folder_pattern));

% Loop through each patient folder
for i = 1:length(patient_folders)
    % Get the current patient folder path
    patient_folder = fullfile(base_path, patient_folders(i).name, '5_MASK', 'NIFTI');

    % Define the input file path for the segmentation mask
    seg_path = fullfile(patient_folder, 'MASK.nii.gz');

    % Check if the MASK.nii.gz file exists
    if isfile(seg_path)
        % Read the original segmentation mask
        seg = niftiread(seg_path);

        % Define a 3D structuring element (sphere-like kernel)
        se = strel('sphere', 3); % Radius of 3 voxels

        % Perform 3D dilation
        seg_dilated = imdilate(seg, se);

        % Perform 3D erosion
        seg_eroded = imerode(seg, se);

        % Compute the hollow volume (difference between dilated and eroded)
        hollow_volume = seg_dilated - seg_eroded;

        % Define the output file path for the periphery mask
        output_path = fullfile(patient_folder, 'PERIPHERY_MASK.nii.gz');

        % Read the original NIfTI metadata to preserve it
        info = niftiinfo(seg_path);

        % Modify the metadata for the new file
        info.Filename = output_path; % Set the new file name
        info.Description = 'Periphery mask (difference between dilation and erosion)';

        % Write the hollow volume to a new NIfTI file
        niftiwrite(hollow_volume, output_path, info);

        % Display progress
        disp(['Processed: ', seg_path, ' -> Saved: ', output_path]);
    else
        % If the MASK.nii.gz file does not exist, warn the user
        disp(['WARNING: MASK.nii.gz not found for ', patient_folders(i).name]);
    end
end

disp('Processing complete.');
