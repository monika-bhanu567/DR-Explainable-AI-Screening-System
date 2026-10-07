import albumentations as A


def get_train_transforms(image_size=224):
    return A.Compose([
        A.Resize(height=image_size, width=image_size),

        # Horizontal flip
        A.HorizontalFlip(p=0.5),

        # Small rotation
        A.Rotate(limit=10, p=0.5),

        # Mild brightness and contrast changes
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.4
        ),

        # Small geometric changes
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(-0.03, 0.03),
            rotate=(-5, 5),
            p=0.3
        ),
    ])


def get_validation_transforms(image_size=224):
    return A.Compose([
        A.Resize(height=image_size, width=image_size)
    ])