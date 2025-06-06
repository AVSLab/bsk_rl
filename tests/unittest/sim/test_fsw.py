from unittest.mock import MagicMock, patch

import numpy as np

from bsk_rl.sim.fsw import BasicFSWModel, FSWModel, ImagingFSWModel, Task, action

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


@patch.multiple(FSWModel, __abstractmethods__=set())
class TestFSWModel:
    def test_base_class(self):
        sat = MagicMock()
        fsw = FSWModel(sat, 1.0)
        # fsw.simulator.CreateNewProcess.assert_called_once()
        assert sat.simulator.world == fsw.world
        assert sat.dynamics == fsw.dynamics

    @patch(module + "base.check_aliveness_checkers", MagicMock(return_value=True))
    def test_is_alive(self):
        fsw = FSWModel(MagicMock(), 1.0)
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
