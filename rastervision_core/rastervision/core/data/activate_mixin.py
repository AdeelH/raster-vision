from abc import abstractmethod


class ActivationError(Exception):
    pass


class ActivateMixin:
    """Defines a mixin for data that can activate and deactivate.

    These methods can open and close files, download files, and do
    whatever has to be done to make the entity usable, and cleanup
    after the entity is not needed anymore.
    """

    _stay_activated = False

    def stay_activated(self):
        self._stay_activated = True

    class ActivateContextManager:
        def __init__(self, activate, deactivate):
            self.activate = activate
            self.deactivate = deactivate

        def __enter__(self):
            self.activate()
            return self

        def __exit__(self, type, value, traceback):
            self.deactivate()

        @classmethod
        def dummy(cls):
            def noop():
                pass

            return cls(noop, noop)

    class CompositeContextManager:
        def __init__(self, *managers):
            self.managers = managers

        def __enter__(self):
            for manager in self.managers:
                manager.__enter__()

        def __exit__(self, type, value, traceback):
            for manager in self.managers:
                manager.__exit__(type, value, traceback)

    def activate(self, stay_activated=False):
        if stay_activated:
            self.stay_activated()

        if hasattr(self, '_mixin_activated') and not self._stay_activated:
            if self._mixin_activated:
                raise ActivationError('This {} is already activated'.format(
                    type(self)))

        def do_activate():
            self._mixin_activated = True
            self._activate()

        def do_deactivate():
            if self._stay_activated:
                return
            self._deactivate()
            self._mixin_activated = False

        manager = ActivateMixin.ActivateContextManager(do_activate,
                                                       do_deactivate)
        subcomponents = self._subcomponents_to_activate()

        if self._stay_activated:
            for c in subcomponents:
                if isinstance(c, ActivateMixin):
                    c.stay_activated()

        if subcomponents:
            manager = ActivateMixin.CompositeContextManager(
                manager, ActivateMixin.compose(*subcomponents))

        return manager

    @abstractmethod
    def _activate(self):
        pass

    @abstractmethod
    def _deactivate(self):
        pass

    def _subcomponents_to_activate(self):
        """Subclasses override this if they have subcomponents
        that may need to be activated when this class is activated
        """
        return []

    @staticmethod
    def with_activation(obj):
        """Method will give activate an object if it mixes in  the ActivateMixin and
        return the context manager, or else return a dummy context manager.
        """
        if obj is None or not isinstance(obj, ActivateMixin):
            return ActivateMixin.dummy()
        else:
            return obj.activate()

    @staticmethod
    def compose(*objs):
        managers = [
            obj.activate() for obj in objs
            if obj is not None and isinstance(obj, ActivateMixin)
        ]
        return ActivateMixin.CompositeContextManager(*managers)
