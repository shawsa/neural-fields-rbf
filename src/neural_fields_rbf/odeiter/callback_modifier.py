def callback_modifier(callback: callable):
    """Some applications require us to modify the current solution before
    producing the next in the sequence. For example, integrating over a delta
    pulse analytically can be done by adding the magnitude of the delta to the
    solution. The motivating example for this however is for domain shifting.
    We simulate pulses on the real line by simulating over a finite portion of
    the real line. If the sim persists long enough then the pulse will
    eventually reach the artificial boundary and the result will be
    non-senical. If we dynamically change the window over which we are
    simulating we can effectively remove this boundary, provided the solution
    remains pulse-like and the width does not grow larger than we expect."""

    def wrapper(cls):
        class SingleStepModifier(cls):
            def update(self, t, u, f, h):
                t, u, f, h = callback(t, u, f, h)
                return super().update(t, u, f, h)

        return SingleStepModifier

    return wrapper
