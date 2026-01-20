import math
import random
import numpy as np
import torch
import parselmouth
from parselmouth.praat import call
import torchaudio.functional as AF
import librosa

try:
    import pyworld
    PYWORLD_AVAILABLE = True
except ImportError:
    PYWORLD_AVAILABLE = False
    print("Warning: pyworld library not found. F0 modification will be limited.")


class BasePerturbation:
    """Base class with shared utility methods."""
    
    @staticmethod
    def wav_to_sound(wav, sampling_frequency: int = 16000):
        """Convert wav array to parselmouth Sound object."""
        if isinstance(wav, parselmouth.Sound):
            return wav
        elif isinstance(wav, np.ndarray):
            return parselmouth.Sound(wav, sampling_frequency=sampling_frequency)
        elif isinstance(wav, list):
            return parselmouth.Sound(np.asarray(wav), sampling_frequency=sampling_frequency)
        elif isinstance(wav, torch.Tensor):
            return parselmouth.Sound(wav.cpu().numpy(), sampling_frequency=sampling_frequency)
        else:
            raise NotImplementedError(f"Unsupported wav type: {type(wav)}")


class SpeakerPerturbation(BasePerturbation):
    """
    Speaker feature perturbation for speaker-invariant learning.
    
    Perturbs speaker-specific characteristics (formants, pitch, EQ)
    while preserving affective information.
    """
    
    @staticmethod
    def get_pitch_median(wav, sr: int = 16000):
        """Get pitch and median pitch value from audio."""
        sound = SpeakerPerturbation.wav_to_sound(wav, sr)
        pitch, pitch_median = None, 0.0
        try:
            pitch = parselmouth.praat.call(sound, "To Pitch", 0.8 / 75, 75, 600)
            pitch_median = parselmouth.praat.call(pitch, "Get quantile", 0.0, 0.0, 0.5, "Hertz")
        except:
            pass
        return pitch, pitch_median
    
    @staticmethod
    def apply_formant_and_pitch_shift(
        sound,
        formant_shift_ratio: float = 1.0,
        pitch_shift_ratio: float = 1.0,
        pitch_range_ratio: float = 1.0,
        duration_factor: float = 1.0
    ):
        """Apply formant and pitch shifting to audio."""
        pitch, new_pitch_median = None, 0.0
        
        if pitch_shift_ratio != 1.0:
            try:
                pitch, pitch_median = SpeakerPerturbation.get_pitch_median(sound, None)
                if pitch_median > 0:
                    new_pitch_median = pitch_median * pitch_shift_ratio
                    pitch_minimum = parselmouth.praat.call(
                        pitch, "Get minimum", 0.0, 0.0, "Hertz", "Parabolic"
                    )
                    scaled_minimum = pitch_minimum * pitch_shift_ratio
                    resulting_minimum = new_pitch_median + (scaled_minimum - new_pitch_median) * pitch_range_ratio
                    
                    if resulting_minimum < 0:
                        new_pitch_median, pitch_range_ratio = 0.0, 1.0
                    if math.isnan(new_pitch_median):
                        new_pitch_median, pitch_range_ratio = 0.0, 1.0
            except:
                pass
        
        try:
            method_name = "Change gender"
            if pitch:
                new_sound = parselmouth.praat.call(
                    (sound, pitch), method_name,
                    formant_shift_ratio, new_pitch_median,
                    pitch_range_ratio, duration_factor
                )
            else:
                new_sound = parselmouth.praat.call(
                    sound, method_name,
                    75, 600, formant_shift_ratio,
                    new_pitch_median, pitch_range_ratio, duration_factor
                )
            return new_sound
        except:
            return sound
    
    @staticmethod
    def power_ratio(r: float, a: float, b: float) -> float:
        """Calculate power ratio for EQ frequency spacing."""
        return a * math.pow((b / a), r)
    
    @staticmethod
    def apply_parametric_equalizer(wav, sr: int, gain_range: tuple = (-12, 12)):
        """Apply parametric equalizer to audio."""
        if isinstance(wav, np.ndarray):
            wav_tensor = torch.from_numpy(wav).float()
        elif isinstance(wav, torch.Tensor):
            wav_tensor = wav.float()
        else:
            raise TypeError("wav must be numpy array or torch tensor")
        
        cutoff_low_freq, cutoff_high_freq, num_filters = 60., 10000., 10
        key_freqs = [
            SpeakerPerturbation.power_ratio(float(z) / num_filters, cutoff_low_freq, cutoff_high_freq)
            for z in range(num_filters)
        ]
        
        try:
            current_device = wav_tensor.device
            
            # Apply high-pass filter
            b_high, a_high = AF.highpass_biquad(sr, key_freqs[0])
            wav_tensor = AF.lfilter(
                wav_tensor.unsqueeze(0),
                torch.tensor(a_high, dtype=wav_tensor.dtype, device=current_device),
                torch.tensor(b_high, dtype=wav_tensor.dtype, device=current_device)
            ).squeeze(0)
            
            # Apply low-pass filter
            b_low, a_low = AF.lowpass_biquad(sr, key_freqs[-1])
            wav_tensor = AF.lfilter(
                wav_tensor.unsqueeze(0),
                torch.tensor(a_low, dtype=wav_tensor.dtype, device=current_device),
                torch.tensor(b_low, dtype=wav_tensor.dtype, device=current_device)
            ).squeeze(0)
            
            # Apply random gain adjustment
            gain_adjustment = 1.0 + random.uniform(-0.1, 0.1)
            wav_tensor = wav_tensor * gain_adjustment
            
            return wav_tensor
        except:
            return wav_tensor
    
    @staticmethod
    def formant_shift(wav_array, sr: int = 16000, formant_range: tuple = (0.7, 1.4)):
        """Apply only formant shifting."""
        sound = SpeakerPerturbation.wav_to_sound(wav_array, sr)
        formant_shifting_ratio = random.uniform(formant_range[0], formant_range[1])
        if random.uniform(-1, 1) > 0:
            formant_shifting_ratio = 1 / formant_shifting_ratio
        
        try:
            sound_new = SpeakerPerturbation.apply_formant_and_pitch_shift(
                sound,
                formant_shift_ratio=formant_shifting_ratio,
                pitch_shift_ratio=1.0,
                pitch_range_ratio=1.0,
                duration_factor=1.0
            )
            return sound_new.values.squeeze()
        except:
            return wav_array
    
    @staticmethod
    def perturb(
        wav_array,
        sr: int = 16000,
        formant_range: tuple = (0.7, 1.4),
        pitch_range: tuple = (0.5, 2.0),
        pitch_range_ratio_range: tuple = (0.7, 1.5),
        gain_range: tuple = (-12, 12)
    ):
        """
        Apply full speaker perturbation (EQ + formant + pitch shifts).
        
        Args:
            wav_array: Input audio array
            sr: Sampling rate
            formant_range: Range for formant shift ratio
            pitch_range: Range for pitch shift ratio
            pitch_range_ratio_range: Range for pitch range ratio
            gain_range: Range for EQ gain adjustment
            
        Returns:
            Audio with perturbed speaker characteristics
        """
        original_device = None
        if isinstance(wav_array, torch.Tensor):
            original_device = wav_array.device
            wav_input = wav_array.clone()
        else:
            wav_input = torch.from_numpy(wav_array.copy()).float()
        
        # Apply parametric equalizer
        try:
            wav_eq_tensor = SpeakerPerturbation.apply_parametric_equalizer(wav_input, sr, gain_range)
            wav_eq = wav_eq_tensor.cpu().numpy() if isinstance(wav_eq_tensor, torch.Tensor) else wav_eq_tensor
        except:
            wav_eq = wav_array.copy() if isinstance(wav_array, np.ndarray) else wav_array.cpu().numpy().copy()
        
        sound = SpeakerPerturbation.wav_to_sound(wav_eq, sr)
        
        # Random formant shift
        formant_shifting_ratio = random.uniform(formant_range[0], formant_range[1])
        if random.uniform(-1, 1) > 0:
            formant_shifting_ratio = 1 / formant_shifting_ratio
        
        # Random pitch shift
        pitch_shift_ratio = random.uniform(pitch_range[0], pitch_range[1])
        if random.uniform(-1, 1) > 0:
            pitch_shift_ratio = 1 / pitch_shift_ratio
        
        # Random pitch range ratio
        pitch_range_ratio = random.uniform(pitch_range_ratio_range[0], pitch_range_ratio_range[1])
        if random.uniform(-1, 1) > 0:
            pitch_range_ratio = 1 / pitch_range_ratio
        
        try:
            sound_new = SpeakerPerturbation.apply_formant_and_pitch_shift(
                sound,
                formant_shift_ratio=formant_shifting_ratio,
                pitch_shift_ratio=pitch_shift_ratio,
                pitch_range_ratio=pitch_range_ratio,
                duration_factor=1.0
            )
            return sound_new.values.squeeze()
        except:
            return wav_array


class EmotionPerturbation(BasePerturbation):
    """
    Information perturbation for affective-invariant learning.
    
    Removes or perturbs affective features while preserving
    speaker identity. Includes:
    - Intensity normalization
    - F0 neutralization (affective)
    """
    
    @staticmethod
    def get_f0_contour(audio_signal, sr: int, time_step: float = 0.01,
                       min_pitch: int = 75, max_pitch: int = 600):
        """Extract F0 contour from audio."""
        sound = EmotionPerturbation.wav_to_sound(audio_signal, sr)
        pitch = call(sound, "To Pitch", time_step, min_pitch, max_pitch)
        
        f0_values = []
        times = []
        for t in pitch.ts():
            f0_val = call(pitch, "Get value at time", t, "Hertz", "Linear")
            if not np.isnan(f0_val) and f0_val > 0:
                f0_values.append(f0_val)
                times.append(t)
        
        return np.array(f0_values), np.array(times), pitch
    
    @staticmethod
    def normalize_intensity(audio_signal, target_rms: float = 0.05):
        """Normalize audio intensity to target RMS."""
        if isinstance(audio_signal, torch.Tensor):
            audio_signal = audio_signal.cpu().numpy()
        
        current_rms = np.sqrt(np.mean(audio_signal ** 2))
        if current_rms <= 1e-10:
            return audio_signal
        
        return audio_signal * (target_rms / current_rms)
    
    @staticmethod
    def remove_affective_features(audio_signal, sr: int, target_rms: float = 0.05):
        """Remove affective features (intensity dynamics, F0 variation)."""
        y = audio_signal.copy() if isinstance(audio_signal, np.ndarray) else audio_signal.cpu().numpy().copy()
        
        # Normalize intensity
        y_intensity_normalized = EmotionPerturbation.normalize_intensity(y, target_rms=target_rms)
        y_f0_modified = y_intensity_normalized
        
        # Neutralize F0 using pyworld if available
        if PYWORLD_AVAILABLE:
            try:
                y_double = y_intensity_normalized.astype(np.double)
                _f0, _time = pyworld.harvest(y_double, sr)
                
                voiced_f0_values = _f0[_f0 > 0]
                if len(voiced_f0_values) > 0:
                    target_f0 = np.mean(voiced_f0_values)
                    sp = pyworld.cheaptrick(y_double, _f0, _time, sr)
                    ap = pyworld.d4c(y_double, _f0, _time, sr)
                    
                    # Set all voiced regions to mean F0
                    f0_neutralized = np.full_like(_f0, target_f0)
                    f0_neutralized[_f0 == 0] = 0
                    
                    y_f0_modified = pyworld.synthesize(f0_neutralized, sp, ap, sr)
                    y_f0_modified = y_f0_modified.astype(np.float32)
                    
                    # Match length
                    if len(y_f0_modified) > len(y):
                        y_f0_modified = y_f0_modified[:len(y)]
                    elif len(y_f0_modified) < len(y):
                        y_f0_modified = np.pad(y_f0_modified, (0, len(y) - len(y_f0_modified)))
            except:
                pass
        else:
            # Fallback: just use intensity-normalized version
            try:
                _ = EmotionPerturbation.get_f0_contour(y_intensity_normalized, sr)
            except:
                pass
        
        return y_f0_modified
    
    @staticmethod
    def perturb(
        wav_array,
        sr: int = 16000,
        perturbation_type: str = "emotion",
        n_bands: int = 10,
        target_rms: float = 0.05
    ):
        """
        Apply information perturbation.
        
        Args:
            wav_array: Input audio array
            sr: Sampling rate
            perturbation_type: "affective" for F0/intensity neutralization
            n_bands: Number of spectral bands
            target_rms: Target RMS for intensity normalization
            
        Returns:
            Perturbed audio array
        """
        return EmotionPerturbation.remove_affective_features(wav_array, sr, target_rms)


def apply_perturbation(
    wav_array,
    sr: int = 16000,
    perturbation_type: str = "speaker",
    **kwargs
):
    """
    Unified interface for applying perturbations.
    
    Args:
        wav_array: Input audio array
        sr: Sampling rate
        perturbation_type: "speaker" or "affective"
        **kwargs: Additional arguments passed to perturbation method
        
    Returns:
        Perturbed audio array
    """
    if perturbation_type == "speaker":
        return SpeakerPerturbation.perturb(
            wav_array, sr,
            formant_range=kwargs.get("formant_range", (0.7, 1.4)),
            pitch_range=kwargs.get("pitch_range", (0.5, 2.0)),
            pitch_range_ratio_range=kwargs.get("pitch_range_ratio_range", (0.7, 1.5)),
            gain_range=kwargs.get("gain_range", (-12, 12))
        )
    elif perturbation_type == "affective":
        return EmotionPerturbation.perturb(
            wav_array, sr,
            perturbation_type=perturbation_type,
            n_bands=kwargs.get("n_bands", 10),
            target_rms=kwargs.get("target_rms", 0.05)
        )
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")