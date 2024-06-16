# Prediction interface for Cog ⚙️
# https://cog.run/python

import csv
import json
import os
import os.path
import shutil
import tempfile

from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        pass

    def predict(
        self,
        audio: Path = Input(description="Speech audio file"),
        checkpoint: str = Input(
            description="Model checkpoint to use. EARS-WHAM speech enhancement or EARS-Reverb dereverberation.",
            option=["EARS-WHAM", "EARS-Reverb"],
        ),
    ) -> Path:

        # Make a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Copy the audio to the temporary directory
                audio_path = os.path.join(temp_dir, os.path.basename(audio))
                print(f"Copying {audio} to {audio_path}")
                shutil.copy(audio, audio_path)

                enhanced_dir = os.path.join(temp_dir, "enhanced")
                os.mkdir(enhanced_dir)

                if checkpoint == "EARS-WHAM":
                    ckpt = "ears_wham.ckpt"
                elif checkpoint == "EARS-Reverb":
                    ckpt = "ears_reverb.ckpt"
                else:
                    raise ValueError(f"Unknown checkpoint: {checkpoint}")

                os.system(
                    f"cd /sgmse ; python3 enhancement.py --test_dir {temp_dir} --enhanced_dir {enhanced_dir} --ckpt {ckpt}"
                )
                files = [
                    f
                    for f in os.listdir(enhanced_dir)
                    if os.path.isfile(os.path.join(enhanced_dir, f))
                    and f.endswith(".wav")
                ]
                assert len(files) == 1
                return files[0]
            except:
                raise
            finally:
                # Delete the temporary directory
                shutil.rmtree(temp_dir)
