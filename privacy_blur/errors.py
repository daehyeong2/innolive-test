class PrivacyBlurError(Exception):
    pass


class UnsupportedFrameTypeError(PrivacyBlurError):
    pass


class PrivacyFilterNotReadyError(PrivacyBlurError):
    pass
