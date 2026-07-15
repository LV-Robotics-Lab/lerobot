# GitHub import record

## Source boundary

[KNOWN] This private repository is a source snapshot of local commit `72d8d0370b395e37bec8d0476372a52ce8ff9120`.

[KNOWN] The snapshot contains two DM/LV hardware-integration commits beyond the common upstream base:

- [KNOWN] `2fcd7f8b82898198b9fff243168403137f02edf9`: RealMan RM75b, uGripper, fisheye, and DM tactile integration.
- [KNOWN] `72d8d0370b395e37bec8d0476372a52ce8ff9120`: AmazingHand support for the bimanual SO follower.

[KNOWN] The common base with the fetched Hugging Face `main` was `2f2b5679510a35aa83fdd8e9f986e134666618bc`; the fetched upstream tip during import was `279c6c7af36183508f5e5c3b2d4bf9c7bb4cc3e6`.

[KNOWN] The full 1,442-commit source history remains in `/Users/boris/Downloads/lerobot`, with the Boris fork and Hugging Face upstream retained as source remotes.

## Why this is a snapshot

[KNOWN] A full-history push was rejected because the inherited history referenced 852 Git LFS objects that were not present in the new organization repository.

[COMPUTED] The inherited history describes about 3.26 GiB of unique LFS content, while the current source tree uses 50 LFS paths backed by 47 unique objects.

[KNOWN] The 47 current objects were uploaded to the private organization repository. Historical test artifacts were not mirrored.

[INFERRED] This snapshot keeps the delivered code and current checkout complete without consuming organization storage for unrelated historical test fixtures. Use the preserved local clone when upstream commit archaeology is required.

## Runtime boundaries

[KNOWN] The RealMan integration expects external RealMan, uGripper, fisheye, and DM Robotics SDK/client components that are not declared or bundled in this snapshot.

[KNOWN] Hardware configuration contains private-network defaults and measured calibration values. No reusable credential was detected in the snapshot, but the values must be reviewed before connecting physical hardware.

[KNOWN] The existing local `.venv` was excluded because it contains stale absolute paths. Recreate the environment from `uv.lock`.

## Licensing and provenance

[KNOWN] The upstream Apache-2.0 license and bundled third-party notices are preserved.

[KNOWN] This import record does not change copyright ownership, grant rights to missing vendor SDKs, or assert that LV Robotics Lab is the canonical LeRobot project.

[INFERRED] Keep the repository private until custom DM/LV components and external SDK dependencies have a written provenance and redistribution record.

