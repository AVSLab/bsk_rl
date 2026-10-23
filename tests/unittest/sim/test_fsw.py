from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from Basilisk.fswAlgorithms import simpleInstrumentController
from Basilisk.utilities import macros

from bsk_rl.sim.fsw import BasicFSWModel, FSWModelABC, ImagingFSWModel, Task, action

module = "bsk_rl.sim.fsw."


def test_action_decorator():
    mock_actions = MagicMock()

    class Object(MagicMock):
        @action
        def action_function(self, foo, bar=2):
            mock_actions(foo, bar=bar)

    o = Object()
    o.tasks = [MagicMock()]
    o.action_function(3, bar=4)
    o.fsw_proc.disableAllTasks.assert_called_once()
    o._zero_gateway_msgs.assert_called_once()
    o.dynamics.reset_for_action.assert_called_once()
    o.tasks[0].reset_for_action.assert_called_once()
    mock_actions.assert_called_once_with(3, bar=4)


@patch.multiple(FSWModelABC, __abstractmethods__=set())
class TestFSWModel:
    def test_base_class(self):
        sat = MagicMock()
        fsw = FSWModelABC(sat, 1.0)
        assert sat.simulator.world == fsw.world
        assert sat.dynamics == fsw.dynamics

    @patch(module + "base.check_aliveness_checkers", MagicMock(return_value=True))
    def test_is_alive(self):
        fsw = FSWModelABC(MagicMock(), 1.0)
        assert fsw.is_alive()


@patch.multiple(Task, __abstractmethods__=set(), name="task")
class TestTask:
    def test_base_class(self):
        fsw = MagicMock()
        task = Task(fsw, 1)
        task.create_task()
        task.fsw.simulator.CreateNewTask.assert_called_once()
        task._setup_fsw_objects()
        task.reset_for_action()
        task.fsw.simulator.disableTask.assert_called_once()


basicfsw = module + "BasicFSWModel."


@patch(basicfsw + "_requires_dyn", MagicMock(return_value=[]))
class TestBasicFSWModel:
    @patch(basicfsw + "_set_messages", MagicMock())
    @patch(basicfsw + "SunPointTask")
    @patch(basicfsw + "NadirPointTask")
    @patch(basicfsw + "RWDesatTask")
    @patch(basicfsw + "TrackingErrorTask")
    @patch(basicfsw + "MRPControlTask")
    def test_make_tasks(self, *args):
        fsw = BasicFSWModel(MagicMock(), 1)
        for task in fsw.tasks:
            task.create_task.assert_called_once()
            task._create_module_data.assert_called_once()
            task._setup_fsw_objects.assert_called_once()


imagingfsw = module + "ImagingFSWModel."


@patch(imagingfsw + "_requires_dyn", MagicMock(return_value=[]))
@patch(imagingfsw + "_make_task_list", MagicMock())
@patch(imagingfsw + "_set_messages", MagicMock())
class TestImagingFSWModel:
    def test_fsw_properties(self):
        fsw = ImagingFSWModel(MagicMock(), 1.0)
        fsw.locPoint = MagicMock(pHat_B=np.array([1.0, 0.0, 0.0]))
        fsw.satellite = MagicMock(dynamics=MagicMock(BP=np.identity(3)))
        assert (fsw.c_hat_P == np.array([1.0, 0.0, 0.0])).all()


class TestImagingTimeTolerance:
    def test_unsupported_controller_logs_warning(self, caplog):
        task = SimpleNamespace(insControl=object())

        ImagingFSWModel.LocPointTask.setup_instrument_time_tolerance(
            task, timeToleranceLower=2.0, timeToleranceUpper=3.0
        )

        assert (
            "insControl.timeTolerance not available; continuing without it"
            in caplog.text
        )

    @pytest.fixture
    def task(self):
        controller = simpleInstrumentController.simpleInstrumentController()
        controller.timeToleranceLower = macros.sec2nano(11)
        controller.timeToleranceUpper = macros.sec2nano(12)
        return SimpleNamespace(insControl=controller)

    @pytest.mark.parametrize(
        "lower,upper", [(0, 0), (0, 2.5), (1.25, 0), (1.25, 2.5), (1e-9, 2e-9)]
    )
    def test_valid_tolerances(self, task, lower, upper):
        ImagingFSWModel.LocPointTask.setup_instrument_time_tolerance(
            task, timeToleranceLower=lower, timeToleranceUpper=upper
        )

        assert task.insControl.timeToleranceLower == macros.sec2nano(lower)
        assert task.insControl.timeToleranceUpper == macros.sec2nano(upper)

    @pytest.mark.parametrize("name", ["timeToleranceLower", "timeToleranceUpper"])
    @pytest.mark.parametrize("value", [-1.0, -1e-9, np.nan, np.inf, -np.inf])
    def test_invalid_tolerance_preserves_both_bounds(self, task, caplog, name, value):
        kwargs = {"timeToleranceLower": 2.0, "timeToleranceUpper": 3.0, name: value}
        bound = "lower" if name == "timeToleranceLower" else "upper"

        with pytest.raises(
            ValueError,
            match=f"{name} must be finite and nonnegative; 0 disables the {bound} time bound",
        ):
            ImagingFSWModel.LocPointTask.setup_instrument_time_tolerance(task, **kwargs)

        assert task.insControl.timeToleranceLower == macros.sec2nano(11)
        assert task.insControl.timeToleranceUpper == macros.sec2nano(12)
        assert not caplog.records
