% zenodo_data_download.m
%
% Description:
%   Download and extract a Zenodo archive containing preprocessed CAMELS
%   .mat files into a specified local data folder. This safe variant:
%     - Skips the download if all expected files already exist.
%     - Wraps websave/unzip in try/catch, deletes the temporary archive on
%       failure, and issues a clear warning with a custom warnID.
%     - Verifies the expected files after extraction and errors if any are
%       missing.
%
% Usage:
%   Edit the variables below as needed, then run the script:
%     zipUrl            - source URL for the Zenodo archive
%     destinationFolder - local folder to extract files into
%   The script will skip downloading when all expected files are present.
%
% Inputs (edit before running):
%   zipUrl              - string: URL to the Zenodo archive (required)
%   destinationFolder   - string: full path to destination folder (required)
%   zipPath             - computed: temporary file path for downloaded archive
%   outDir              - computed: output directory (same as destinationFolder)
%
% Expected Output / Side Effects:
%   - Extracted files appear in destinationFolder. Expected files include:
%       camels_421_partitioning.mat
%       camels_421_inputs.mat
%       camels_421_target_deterministic.mat
%       camels_421_target_hurdle_rg.mat
%   - filesExtracted : cell array returned by unzip (if successful)
%   - Prints progress messages to the command window.
%   - On download/unzip failure, the temporary archive (zipPath) is deleted
%     and a warning is raised with warnID "zenodo:downloadFailed"; the
%     script returns early (change to rethrow(ME) to propagate the error).
%   - If expected files are missing after extraction the script calls error()
%     to abort further processing.
%
% Behavior and Notes:
%   - Uses websave with weboptions('Timeout',N). Increase Timeout if needed.
%   - If destinationFolder does not exist, the script creates it.
%   - The temporary archive at zipPath is overwritten on each run.
%   - To change failure handling: replace return with rethrow(ME) in the
%     catch block to surface the exception.
%   - For integration into larger pipelines, consider returning status codes
%     or throwing exceptions instead of returning silently.
%
% Recommended Improvements (optional):
%   - Add logging to a file instead of only printing to the console.
%   - Add checksum verification of extracted files when available.
%   - Make expectedFiles (and their source paths) configurable inputs.
%
% Author:
%   John Quilty
%
% Date:
%   2026-01-22
%
% License:
%   MIT License (see LICENSE file)
%
% -------------------------------------------------------------------------


zipUrl = "https://zenodo.org/api/records/18332982/files-archive";
destinationFolder = "F:\projects\Paper_HYDROL74788\semicontinuousDL_camelsUSA\data";
zipPath = fullfile(tempdir,"archive.zip");
outDir = destinationFolder;

% list of expected files after unzip
expectedFiles = { ...
    fullfile(outDir,"camels_421_partitioning.mat"), ...
    fullfile(outDir,"camels_421_inputs.mat"), ...
    fullfile(outDir,"camels_421_target_deterministic.mat"), ...
    fullfile(outDir,"camels_421_target_hurdle_rg.mat") ...
    };

% create destination if needed
if ~exist(outDir,'dir')
    mkdir(outDir);
end

% Optionally skip download if all expected files already exist
allExist = all(cellfun(@isfile, expectedFiles));
if allExist
    fprintf("All expected files already exist in '%s'. Skipping download.\n", outDir);
else
    opts = weboptions('Timeout',60); % increase timeout as needed

    % Wrap download+unzip in try/catch to avoid partial extractions
    try
        fprintf("Downloading archive to %s ...\n", zipPath);
        websave(zipPath, zipUrl, opts);

        fprintf("Unzipping archive to %s ...\n", outDir);
        filesExtracted = unzip(zipPath, outDir); % returns cell array of extracted filenames

        fprintf("Download and unzip completed. Extracted %d files.\n", numel(filesExtracted));
    catch ME
        % Clean up partially downloaded archive if present
        if isfile(zipPath)
            try
                delete(zipPath);
            catch
                % ignore delete errors
            end
        end

        % Build a clear message and use a custom warnID to avoid ambiguous syntax
        warnID = "zenodo:downloadFailed";
        msg = sprintf("Download/unzip failed: %s\n\nFull report:\n%s", ME.message, ME.getReport());
        warning(warnID, msg);

        % Stop further execution (change to rethrow(ME) if you want to propagate the error)
        return;
    end
end

% Verify expected files exist and report missing ones
missing = expectedFiles(~cellfun(@isfile, expectedFiles));
if isempty(missing)
    fprintf("All expected files are present in '%s'.\n", outDir);
else
    fprintf("Missing %d expected file(s) in '%s':\n", numel(missing), outDir);
    for k = 1:numel(missing)
        fprintf("  - %s\n", missing{k});
    end
    error("Required data files missing. Aborting.");
end
