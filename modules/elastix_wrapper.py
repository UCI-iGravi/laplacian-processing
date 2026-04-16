import os
import subprocess
from pathlib import Path

import SimpleITK as sitk
import numpy as np


class ElastixWrapper:
    def __init__(self, elastix_path="elastix/elastix.exe", transformix_path="elastix/transformix.exe"):
        """
        elastix_path: Path to elastix binary (e.g., 'C:/elastix/bin/elastix.exe')
        transformix_path: Path to transformix binary
        """
        self.elastix_path = str(elastix_path)
        self.transformix_path = str(transformix_path)

    @staticmethod
    def numpy_to_image(array, path):
        image = sitk.GetImageFromArray(array.astype(np.float32))
        sitk.WriteImage(image, str(path))

    @staticmethod
    def image_to_numpy(path):
        image = sitk.ReadImage(str(path))
        return sitk.GetArrayFromImage(image)

    def register(self, fixed_array, moving_array, param_files, output_dir="elastix_output"):
        """
        Run elastix registration.
        param_files: list of parameter files (e.g., ["rigid.txt", "bspline.txt"])
        """
        os.makedirs(output_dir, exist_ok=True)

        fixed_path = Path(output_dir) / "fixed.nii.gz"
        moving_path = Path(output_dir) / "moving.nii.gz"

        self.numpy_to_image(fixed_array, fixed_path)
        self.numpy_to_image(moving_array, moving_path)

        command = [
            self.elastix_path,
            "-f", str(fixed_path),
            "-m", str(moving_path),
            "-out", str(output_dir),
        ]

        for p in param_files:
            command += ["-p", str(p)]

        print("[INFO] Running:", " ".join(command))
        subprocess.run(command, check=True)

        result_path = Path(output_dir) / "result.0.nii.gz"
        return self.image_to_numpy(result_path)

    def get_deformation_field(self, output_dir="elastix_output"):
        transform_path = Path(output_dir) / "TransformParameters.1.txt"

        fixed_img = sitk.ReadImage(Path(output_dir) / "fixed.nii.gz")
        size = fixed_img.GetSize()
        spacing = fixed_img.GetSpacing()

        with open(transform_path, "r") as f:
            lines = f.readlines()

        def has_tag(tag):
            return any(tag in line for line in lines)

        if not has_tag("(Size"):
            lines.append(f'(Size {" ".join(str(i) for i in size)})\n')
        if not has_tag("(Spacing"):
            lines.append(f'(Spacing {" ".join(str(s) for s in spacing)})\n')
        if not has_tag("(Origin"):
            origin = fixed_img.GetOrigin()
            lines.append(f'(Origin {" ".join(str(o) for o in origin)})\n')
        if not has_tag("(ComputeDeformationField"):
            lines.append('(ComputeDeformationField "true")\n')

        with open(transform_path, "w") as f:
            f.writelines(lines)

        command = [
            self.transformix_path,
            "-def", "all",
            "-out", str(output_dir),
            "-tp", str(transform_path),
        ]
        print("[INFO] Running:", " ".join(command))
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(result.stdout)

        deformation_field = Path(output_dir) / "deformationField.mhd"
        if not deformation_field.exists():
            raise FileNotFoundError("Transformix did not generate deformationField.mhd")

        return self.image_to_numpy(deformation_field)

    def apply_transform(self, array, output_dir="elastix_output", result_name="warped.nii.gz"):
        """Warp another image using the computed transformation."""
        os.makedirs(output_dir, exist_ok=True)
        moving_path = Path(output_dir) / "temp_moving.nii.gz"
        self.numpy_to_image(array, moving_path)

        command = [
            self.transformix_path,
            "-in", str(moving_path),
            "-out", str(output_dir),
            "-tp", str(Path(output_dir) / "TransformParameters.1.txt"),
        ]
        print("[INFO] Running:", " ".join(command))
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(result.stdout)

        result_path = Path(output_dir) / "result.nii.gz"
        return self.image_to_numpy(result_path)
