#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on juin 24, 2024, at 17:14
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'mSST_v15_german_LSL'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s_%s' % (expInfo['participant'], expInfo['session'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Juliette\\OneDrive\\Documents\\PhD\\Psychopy\\modifiedStopSignalTask\\mSST_v15_german_LSL.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1920, 1080], fullscr=True, screen=1,
            winType='pyglet', allowStencil=True,
            monitor='testMonitor', color='black', colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = 'black'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "welcome_screen" ---
    welcome_text = visual.TextStim(win=win, name='welcome_text',
        text='Herzlich willkommen!\n\n\n\nWenn Sie Rechtshänder sind, drücken Sie bitte die rechte Taste.\n\n\nWenn Sie Linkshänder sind, drücken Sie bitte die linke Taste.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    right_or_left_resp = keyboard.Keyboard()
    # Run 'Begin Experiment' code from initialize_LSL_streams
    from pylsl import StreamInfo, StreamOutlet
    
    # Set up LabStreamingLayer stream.
    psychopy_info = StreamInfo(name='Psychopy', type='Markers', channel_count=1, nominal_srate=0, channel_format='string', source_id='psy_marker')
    psychopy_outlet = StreamOutlet(psychopy_info)  # Broadcast the stream.
    
    
    # --- Initialize components for Routine "Task_presentation" ---
    task_presentation_text = visual.TextStim(win=win, name='task_presentation_text',
        text='In diesem Experiment werden Sie eine modifizierte Stoppsignal-Aufgabe durchführen.\n\nEs gibt 4 Bedingungen, die Ihnen auf dem nächsten Bildschirm präsentiert werden.\nDas Ziel ist es, so schnell wie möglich die Taste zu drücken, wenn Sie ein "Go"-Signal sehen, aber diese Bewegung zu unterlassen, wenn ein "Stop"-Signal nach diesem "Go"-Signal erscheint. \n\nSchnelles Drücken bei einem "Go"-Signal ist genauso wichtig wie das Drücken zu vermeiden bei einem "Stop"-Signal.\n\nDrücken Sie die Taste, um fortzufahren und die verschiedenen Bedingungen zu sehen.',
        font='Calibri',
        pos=(0, 0), height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    presentation_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Condition_presentation" ---
    condition_image = visual.ImageStim(
        win=win,
        name='condition_image', 
        image='conditions_german.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.8, 0.8),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    condition_response = keyboard.Keyboard()
    
    # --- Initialize components for Routine "trial_randomization_training" ---
    
    # --- Initialize components for Routine "fixation_period_training" ---
    fixation_cross_training = visual.ShapeStim(
        win=win, name='fixation_cross_training', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    early_press_resp_training = keyboard.Keyboard()
    
    # --- Initialize components for Routine "trial_training" ---
    # Run 'Begin Experiment' code from trial_probability_training
    first_stop_trial_training = True
    
    ssd_training_list = []
    go_rectangle_training = visual.Rect(
        win=win, name='go_rectangle_training',
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=1.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    stop_signal_triangle_training = visual.ShapeStim(
        win=win, name='stop_signal_triangle_training',
        size=[1.0, 1.0], vertices='triangle',
        ori=1.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    key_resp_training = keyboard.Keyboard()
    
    # --- Initialize components for Routine "blank_training" ---
    no_shape_2 = visual.ShapeStim(
        win=win, name='no_shape_2',
        size=(0.5, 0.5), vertices='triangle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "feedback_accuracy_training" ---
    # Run 'Begin Experiment' code from update_scores_training
    msg = ''
    total_score_training = 0
    feedback_text_training = visual.TextStim(win=win, name='feedback_text_training',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "determine_start_ssd" ---
    
    # --- Initialize components for Routine "ssd_screen" ---
    
    # --- Initialize components for Routine "Start_real_task" ---
    text_start_task = visual.TextStim(win=win, name='text_start_task',
        text='Herzlichen Glückwunsch, Sie haben die Trainingsphase erfolgreich beendet!\n \n\n\nBitte geben Sie der Versuchsleiterin Bescheid, um fortzufahren.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_start_main_task = keyboard.Keyboard()
    
    # --- Initialize components for Routine "self_paced_start" ---
    self_paced_start_text = visual.TextStim(win=win, name='self_paced_start_text',
        text='Beginn der Aufgabe. \n\nDrücken Sie zum Starten die rechte Taste.\n\nViel Erfolg!\n',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_self_paced_start = keyboard.Keyboard()
    
    # --- Initialize components for Routine "trial_randomization" ---
    # Run 'Begin Experiment' code from randomization_block
    HALF_BLOCK = False
    
    # --- Initialize components for Routine "fixation_period" ---
    # Run 'Begin Experiment' code from code_fixation_cross
    early_press = False
    fixation_cross = visual.ShapeStim(
        win=win, name='fixation_cross', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    early_press_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "trial" ---
    # Run 'Begin Experiment' code from trial_probability
    first_stop_trial = True
    go_rectangle = visual.Rect(
        win=win, name='go_rectangle',
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=1.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    stop_signal_triangle = visual.ShapeStim(
        win=win, name='stop_signal_triangle',
        size=[1.0, 1.0], vertices='triangle',
        ori=1.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    key_resp_experiment = keyboard.Keyboard()
    
    # --- Initialize components for Routine "blank" ---
    no_shape = visual.ShapeStim(
        win=win, name='no_shape',
        size=(0.5, 0.5), vertices='triangle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    late_key_resp1 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "feedback_accuracy" ---
    # Run 'Begin Experiment' code from update_scores
    msg = ''
    total_score = 0
    total_nb_success_stop = 0
    total_nb_unsuccess_stop = 0
    feedback_text = visual.TextStim(win=win, name='feedback_text',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    late_key_resp2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "find_nb_each_stop_trials" ---
    # Run 'Begin Experiment' code from sum_success_unsuccess
    msg_stops = ""
    break_rep = 1
    
    # --- Initialize components for Routine "break_between_block" ---
    break1_text = visual.TextStim(win=win, name='break1_text',
        text='Kurze Pause! \n\nSie können sich jetzt ausruhen, bevor Sie wieder anfangen.\n\nWenn Sie bereit sind, drücken Sie die Taste, um mit dem nächsten Block fortzufahren.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    break1_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "end_screen" ---
    end_text = visual.TextStim(win=win, name='end_text',
        text='Herzlichen Glückwunsch, Sie haben die Aufgabe erfolgreich beendet!\n \n\n\nBitte geben Sie der Versuchsleiterin Bescheid.\n \n\n\nDanke für die Teilnahme.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    end_resp = keyboard.Keyboard()
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "welcome_screen" ---
    continueRoutine = True
    # update component parameters for each repeat
    right_or_left_resp.keys = []
    right_or_left_resp.rt = []
    _right_or_left_resp_allKeys = []
    # keep track of which components have finished
    welcome_screenComponents = [welcome_text, right_or_left_resp]
    for thisComponent in welcome_screenComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcome_screen" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welcome_text* updates
        
        # if welcome_text is starting this frame...
        if welcome_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_text.frameNStart = frameN  # exact frame index
            welcome_text.tStart = t  # local t and not account for scr refresh
            welcome_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_text.started')
            # update status
            welcome_text.status = STARTED
            welcome_text.setAutoDraw(True)
        
        # if welcome_text is active this frame...
        if welcome_text.status == STARTED:
            # update params
            pass
        
        # *right_or_left_resp* updates
        waitOnFlip = False
        
        # if right_or_left_resp is starting this frame...
        if right_or_left_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right_or_left_resp.frameNStart = frameN  # exact frame index
            right_or_left_resp.tStart = t  # local t and not account for scr refresh
            right_or_left_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right_or_left_resp, 'tStartRefresh')  # time at next scr refresh
            # update status
            right_or_left_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(right_or_left_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(right_or_left_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if right_or_left_resp.status == STARTED and not waitOnFlip:
            theseKeys = right_or_left_resp.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
            _right_or_left_resp_allKeys.extend(theseKeys)
            if len(_right_or_left_resp_allKeys):
                right_or_left_resp.keys = _right_or_left_resp_allKeys[-1].name  # just the last key pressed
                right_or_left_resp.rt = _right_or_left_resp_allKeys[-1].rt
                right_or_left_resp.duration = _right_or_left_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcome_screenComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome_screen" ---
    for thisComponent in welcome_screenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if right_or_left_resp.keys in ['', [], None]:  # No response was made
        right_or_left_resp.keys = None
    thisExp.addData('right_or_left_resp.keys',right_or_left_resp.keys)
    if right_or_left_resp.keys != None:  # we had a response
        thisExp.addData('right_or_left_resp.rt', right_or_left_resp.rt)
        thisExp.addData('right_or_left_resp.duration', right_or_left_resp.duration)
    thisExp.nextEntry()
    # Run 'End Routine' code from determine_which_button
    right_or_left = right_or_left_resp.keys
    
    # the Routine "welcome_screen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    show_rules_again = data.TrialHandler(nReps=0.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='show_rules_again')
    thisExp.addLoop(show_rules_again)  # add the loop to the experiment
    thisShow_rules_again = show_rules_again.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisShow_rules_again.rgb)
    if thisShow_rules_again != None:
        for paramName in thisShow_rules_again:
            globals()[paramName] = thisShow_rules_again[paramName]
    
    for thisShow_rules_again in show_rules_again:
        currentLoop = show_rules_again
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisShow_rules_again.rgb)
        if thisShow_rules_again != None:
            for paramName in thisShow_rules_again:
                globals()[paramName] = thisShow_rules_again[paramName]
        
        # --- Prepare to start Routine "Task_presentation" ---
        continueRoutine = True
        # update component parameters for each repeat
        presentation_resp.keys = []
        presentation_resp.rt = []
        _presentation_resp_allKeys = []
        # keep track of which components have finished
        Task_presentationComponents = [task_presentation_text, presentation_resp]
        for thisComponent in Task_presentationComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Task_presentation" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *task_presentation_text* updates
            
            # if task_presentation_text is starting this frame...
            if task_presentation_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                task_presentation_text.frameNStart = frameN  # exact frame index
                task_presentation_text.tStart = t  # local t and not account for scr refresh
                task_presentation_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(task_presentation_text, 'tStartRefresh')  # time at next scr refresh
                # update status
                task_presentation_text.status = STARTED
                task_presentation_text.setAutoDraw(True)
            
            # if task_presentation_text is active this frame...
            if task_presentation_text.status == STARTED:
                # update params
                pass
            
            # *presentation_resp* updates
            waitOnFlip = False
            
            # if presentation_resp is starting this frame...
            if presentation_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                presentation_resp.frameNStart = frameN  # exact frame index
                presentation_resp.tStart = t  # local t and not account for scr refresh
                presentation_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(presentation_resp, 'tStartRefresh')  # time at next scr refresh
                # update status
                presentation_resp.status = STARTED
                # AllowedKeys looks like a variable named `right_or_left`
                if not type(right_or_left) in [list, tuple, np.ndarray]:
                    if not isinstance(right_or_left, str):
                        logging.error('AllowedKeys variable `right_or_left` is not string- or list-like.')
                        core.quit()
                    elif not ',' in right_or_left:
                        right_or_left = (right_or_left,)
                    else:
                        right_or_left = eval(right_or_left)
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(presentation_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(presentation_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if presentation_resp.status == STARTED and not waitOnFlip:
                theseKeys = presentation_resp.getKeys(keyList=list(right_or_left), ignoreKeys=["escape"], waitRelease=False)
                _presentation_resp_allKeys.extend(theseKeys)
                if len(_presentation_resp_allKeys):
                    presentation_resp.keys = _presentation_resp_allKeys[-1].name  # just the last key pressed
                    presentation_resp.rt = _presentation_resp_allKeys[-1].rt
                    presentation_resp.duration = _presentation_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Task_presentationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Task_presentation" ---
        for thisComponent in Task_presentationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # the Routine "Task_presentation" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Condition_presentation" ---
        continueRoutine = True
        # update component parameters for each repeat
        condition_response.keys = []
        condition_response.rt = []
        _condition_response_allKeys = []
        # keep track of which components have finished
        Condition_presentationComponents = [condition_image, condition_response]
        for thisComponent in Condition_presentationComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Condition_presentation" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *condition_image* updates
            
            # if condition_image is starting this frame...
            if condition_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                condition_image.frameNStart = frameN  # exact frame index
                condition_image.tStart = t  # local t and not account for scr refresh
                condition_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(condition_image, 'tStartRefresh')  # time at next scr refresh
                # update status
                condition_image.status = STARTED
                condition_image.setAutoDraw(True)
            
            # if condition_image is active this frame...
            if condition_image.status == STARTED:
                # update params
                pass
            
            # *condition_response* updates
            waitOnFlip = False
            
            # if condition_response is starting this frame...
            if condition_response.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                condition_response.frameNStart = frameN  # exact frame index
                condition_response.tStart = t  # local t and not account for scr refresh
                condition_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(condition_response, 'tStartRefresh')  # time at next scr refresh
                # update status
                condition_response.status = STARTED
                # AllowedKeys looks like a variable named `right_or_left`
                if not type(right_or_left) in [list, tuple, np.ndarray]:
                    if not isinstance(right_or_left, str):
                        logging.error('AllowedKeys variable `right_or_left` is not string- or list-like.')
                        core.quit()
                    elif not ',' in right_or_left:
                        right_or_left = (right_or_left,)
                    else:
                        right_or_left = eval(right_or_left)
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(condition_response.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(condition_response.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if condition_response.status == STARTED and not waitOnFlip:
                theseKeys = condition_response.getKeys(keyList=list(right_or_left), ignoreKeys=["escape"], waitRelease=False)
                _condition_response_allKeys.extend(theseKeys)
                if len(_condition_response_allKeys):
                    condition_response.keys = _condition_response_allKeys[-1].name  # just the last key pressed
                    condition_response.rt = _condition_response_allKeys[-1].rt
                    condition_response.duration = _condition_response_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Condition_presentationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Condition_presentation" ---
        for thisComponent in Condition_presentationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # the Routine "Condition_presentation" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 0.0 repeats of 'show_rules_again'
    
    
    # --- Prepare to start Routine "trial_randomization_training" ---
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from randomization_block_2
    import random
    
    n_training_trials = 36
    sequences_length = 6
    # my values
    """
    percentage_GT = 0.5
    percentage_GF = 0.17
    percentage_GC = 0.17
    percentage_ST = 0.17
    """
    # try something more similar to Obeso:
    percentage_GT = 0.34
    percentage_GF = 0.34
    percentage_GC = 0.17
    percentage_ST = 0.17
    
    n_GT = int(sequences_length * percentage_GT)
    n_GF = int(sequences_length * percentage_GF)
    n_GC = int(sequences_length * percentage_GC)
    n_ST = int(sequences_length * percentage_ST)
    
    sequences_number = int(n_training_trials / sequences_length)
    training_conditions = []
    sequence = []
    sequence_without_stop = []
    following_sequences = []
    trial_names = ['go_trial', 'go_continue_trial', 'go_fast_trial', 'stop_trial']
    
    # Generate a sequence
    sequence = (
        ['go_trial'] * int(n_GT) +
        ['go_continue_trial'] * int(n_GC) +
        ['go_fast_trial'] * int(n_GF) +
        ['stop_trial'] * int(n_ST)
    )
    
    sequence_without_stop = (
        ['go_trial'] * int(n_GT) +
        ['go_continue_trial'] * int(n_GC) +
        ['go_fast_trial'] * int(n_GF)
    )
    
    # Shuffle the sequence initially
    random.shuffle(sequence)
    random.shuffle(sequence_without_stop)
    stop_trial = (['stop_trial'])
    
    following_sequences.extend(sequence_without_stop)
    following_sequences.extend(stop_trial)
    following_sequences.extend(stop_trial)
    random.shuffle(sequence_without_stop)
    following_sequences.extend(sequence_without_stop)
    
    
    random_index = random.randint(0, sequences_number - 2)
    random_index
    
    for i in range(0, sequences_number-1, 1):
        if i == random_index:
            training_conditions.extend(following_sequences)
            continue
        if training_conditions and training_conditions[-1] == 'stop_trial' and sequence[0] == 'stop_trial':
            while sequence[0] == 'stop_trial':
                random.shuffle(sequence)
            training_conditions.extend(sequence)
            random.shuffle(sequence)
        else:
            training_conditions.extend(sequence)
            random.shuffle(sequence)
    
    # keep track of which components have finished
    trial_randomization_trainingComponents = []
    for thisComponent in trial_randomization_trainingComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "trial_randomization_training" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trial_randomization_trainingComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "trial_randomization_training" ---
    for thisComponent in trial_randomization_trainingComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "trial_randomization_training" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trial_loop_training = data.TrialHandler(nReps=n_training_trials, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trial_loop_training')
    thisExp.addLoop(trial_loop_training)  # add the loop to the experiment
    thisTrial_loop_training = trial_loop_training.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_loop_training.rgb)
    if thisTrial_loop_training != None:
        for paramName in thisTrial_loop_training:
            globals()[paramName] = thisTrial_loop_training[paramName]
    
    for thisTrial_loop_training in trial_loop_training:
        currentLoop = trial_loop_training
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_loop_training.rgb)
        if thisTrial_loop_training != None:
            for paramName in thisTrial_loop_training:
                globals()[paramName] = thisTrial_loop_training[paramName]
        
        # --- Prepare to start Routine "fixation_period_training" ---
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_fixation_cross_training
        time_fixation_training = np.random.uniform(0.5,1.5)
        thisExp.addData('time_fixation_training', time_fixation_training)
        
        if right_or_left_resp.keys == 'right':
            key_resp = 'right'
        elif right_or_left_resp.keys == 'left':
            key_resp = 'left'
        
        early_press_resp_training.keys = []
        early_press_resp_training.rt = []
        _early_press_resp_training_allKeys = []
        # keep track of which components have finished
        fixation_period_trainingComponents = [fixation_cross_training, early_press_resp_training]
        for thisComponent in fixation_period_trainingComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation_period_training" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation_cross_training* updates
            
            # if fixation_cross_training is starting this frame...
            if fixation_cross_training.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_cross_training.frameNStart = frameN  # exact frame index
                fixation_cross_training.tStart = t  # local t and not account for scr refresh
                fixation_cross_training.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_cross_training, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_cross_training.started')
                # update status
                fixation_cross_training.status = STARTED
                fixation_cross_training.setAutoDraw(True)
            
            # if fixation_cross_training is active this frame...
            if fixation_cross_training.status == STARTED:
                # update params
                pass
            
            # if fixation_cross_training is stopping this frame...
            if fixation_cross_training.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_cross_training.tStartRefresh + time_fixation_training-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_cross_training.tStop = t  # not accounting for scr refresh
                    fixation_cross_training.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_cross_training.stopped')
                    # update status
                    fixation_cross_training.status = FINISHED
                    fixation_cross_training.setAutoDraw(False)
            
            # *early_press_resp_training* updates
            waitOnFlip = False
            
            # if early_press_resp_training is starting this frame...
            if early_press_resp_training.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                early_press_resp_training.frameNStart = frameN  # exact frame index
                early_press_resp_training.tStart = t  # local t and not account for scr refresh
                early_press_resp_training.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(early_press_resp_training, 'tStartRefresh')  # time at next scr refresh
                # update status
                early_press_resp_training.status = STARTED
                # AllowedKeys looks like a variable named `right_or_left`
                if not type(right_or_left) in [list, tuple, np.ndarray]:
                    if not isinstance(right_or_left, str):
                        logging.error('AllowedKeys variable `right_or_left` is not string- or list-like.')
                        core.quit()
                    elif not ',' in right_or_left:
                        right_or_left = (right_or_left,)
                    else:
                        right_or_left = eval(right_or_left)
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(early_press_resp_training.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(early_press_resp_training.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if early_press_resp_training is stopping this frame...
            if early_press_resp_training.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > early_press_resp_training.tStartRefresh + time_fixation_training-frameTolerance:
                    # keep track of stop time/frame for later
                    early_press_resp_training.tStop = t  # not accounting for scr refresh
                    early_press_resp_training.frameNStop = frameN  # exact frame index
                    # update status
                    early_press_resp_training.status = FINISHED
                    early_press_resp_training.status = FINISHED
            if early_press_resp_training.status == STARTED and not waitOnFlip:
                theseKeys = early_press_resp_training.getKeys(keyList=list(right_or_left), ignoreKeys=["escape"], waitRelease=False)
                _early_press_resp_training_allKeys.extend(theseKeys)
                if len(_early_press_resp_training_allKeys):
                    early_press_resp_training.keys = _early_press_resp_training_allKeys[-1].name  # just the last key pressed
                    early_press_resp_training.rt = _early_press_resp_training_allKeys[-1].rt
                    early_press_resp_training.duration = _early_press_resp_training_allKeys[-1].duration
                    # was this correct?
                    if (early_press_resp_training.keys == str(key_resp)) or (early_press_resp_training.keys == key_resp):
                        early_press_resp_training.corr = 1
                    else:
                        early_press_resp_training.corr = 0
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixation_period_trainingComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation_period_training" ---
        for thisComponent in fixation_period_trainingComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # check responses
        if early_press_resp_training.keys in ['', [], None]:  # No response was made
            early_press_resp_training.keys = None
            # was no response the correct answer?!
            if str(key_resp).lower() == 'none':
               early_press_resp_training.corr = 1;  # correct non-response
            else:
               early_press_resp_training.corr = 0;  # failed to respond (incorrectly)
        # store data for trial_loop_training (TrialHandler)
        trial_loop_training.addData('early_press_resp_training.keys',early_press_resp_training.keys)
        trial_loop_training.addData('early_press_resp_training.corr', early_press_resp_training.corr)
        if early_press_resp_training.keys != None:  # we had a response
            trial_loop_training.addData('early_press_resp_training.rt', early_press_resp_training.rt)
            trial_loop_training.addData('early_press_resp_training.duration', early_press_resp_training.duration)
        # the Routine "fixation_period_training" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trial_training" ---
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from trial_probability_training
        # Begin Routine tab in the Code Component
        current_training_trial_index = trial_loop_training.thisN  # Get the current iteration index
        condition = training_conditions[current_training_trial_index]
        
        color_go = 'white'
        color_stop = 'white'
        
        # initialize ssd value:
        if (first_stop_trial_training and condition == 'stop_trial'):
            ssd = 0.2
            first_stop_trial_training = False
        elif (first_stop_trial_training and condition == 'go_continue_trial'):
            last_ssd = 0.2
            trial_type = 'go_continue_trial'
            corr_resp = key_resp
            color_go = 'white'
            color_stop = 'blue'
            stop_signal_time = last_ssd
            go_trial_rectangle_duration = stop_signal_time
            # end_routine = stop_signal_time + 0.5
            end_routine = stop_signal_time + 1
            thisExp.addData('continue_signal_time',stop_signal_time)
            thisExp.addData('trial_type',trial_type)
            thisExp.addData('corr_resp',corr_resp)
        
        
        if not first_stop_trial_training:
            if condition == 'go_continue_trial':
                trial_type = 'go_continue_trial'
                corr_resp = key_resp
                color_go = 'white'
                color_stop = 'blue'
                stop_signal_time = last_ssd
                go_trial_rectangle_duration = stop_signal_time
                end_routine = stop_signal_time + 1
                # end_routine = stop_signal_time + 0.5
                thisExp.addData('continue_signal_time',stop_signal_time)
                thisExp.addData('trial_type',trial_type)
                thisExp.addData('corr_resp',corr_resp)
            elif condition == 'stop_trial':
                trial_type = 'stop_trial'
                corr_resp = None
                color_go = 'white'
                color_stop = 'white'
                stop_signal_time = ssd
                end_routine = stop_signal_time + 1
                # end_routine = stop_signal_time + 0.5
                go_trial_rectangle_duration = stop_signal_time
                thisExp.addData('trial_type',trial_type)
                thisExp.addData('corr_resp',corr_resp)
                thisExp.addData('ssd_training',stop_signal_time)
                ssd_training_list.append(stop_signal_time)
        
        if condition == 'go_fast_trial': 
            trial_type = 'go_fast_trial'
            corr_resp = key_resp
            color_go = 'blue'
            go_trial_rectangle_duration = 1
            stop_signal_time = 3.5
            end_routine = go_trial_rectangle_duration
            thisExp.addData('trial_type',trial_type)
            thisExp.addData('corr_resp',corr_resp)
        elif condition == 'go_trial': 
            trial_type = 'go_trial'
            corr_resp = key_resp
            color_go = 'white'
            go_trial_rectangle_duration = 1
            end_routine = go_trial_rectangle_duration
            stop_signal_time = 3.5
            thisExp.addData('trial_type',trial_type)
            thisExp.addData('corr_resp',corr_resp)
        go_rectangle_training.setFillColor(color_go)
        go_rectangle_training.setPos((0, 0))
        go_rectangle_training.setSize((0.5, 0.5))
        go_rectangle_training.setOri(0.0)
        stop_signal_triangle_training.setFillColor(color_stop)
        stop_signal_triangle_training.setPos((0, 0))
        stop_signal_triangle_training.setSize((0.5, 0.5))
        stop_signal_triangle_training.setOri(0.0)
        key_resp_training.keys = []
        key_resp_training.rt = []
        _key_resp_training_allKeys = []
        # keep track of which components have finished
        trial_trainingComponents = [go_rectangle_training, stop_signal_triangle_training, key_resp_training]
        for thisComponent in trial_trainingComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_training" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > end_routine-frameTolerance:
                continueRoutine = False
            
            # *go_rectangle_training* updates
            
            # if go_rectangle_training is starting this frame...
            if go_rectangle_training.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                go_rectangle_training.frameNStart = frameN  # exact frame index
                go_rectangle_training.tStart = t  # local t and not account for scr refresh
                go_rectangle_training.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(go_rectangle_training, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'go_rectangle_training.started')
                # update status
                go_rectangle_training.status = STARTED
                go_rectangle_training.setAutoDraw(True)
            
            # if go_rectangle_training is active this frame...
            if go_rectangle_training.status == STARTED:
                # update params
                pass
            
            # if go_rectangle_training is stopping this frame...
            if go_rectangle_training.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > go_rectangle_training.tStartRefresh + go_trial_rectangle_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    go_rectangle_training.tStop = t  # not accounting for scr refresh
                    go_rectangle_training.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'go_rectangle_training.stopped')
                    # update status
                    go_rectangle_training.status = FINISHED
                    go_rectangle_training.setAutoDraw(False)
            
            # *stop_signal_triangle_training* updates
            
            # if stop_signal_triangle_training is starting this frame...
            if stop_signal_triangle_training.status == NOT_STARTED and tThisFlip >= stop_signal_time-frameTolerance:
                # keep track of start time/frame for later
                stop_signal_triangle_training.frameNStart = frameN  # exact frame index
                stop_signal_triangle_training.tStart = t  # local t and not account for scr refresh
                stop_signal_triangle_training.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(stop_signal_triangle_training, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'stop_signal_triangle_training.started')
                # update status
                stop_signal_triangle_training.status = STARTED
                stop_signal_triangle_training.setAutoDraw(True)
            
            # if stop_signal_triangle_training is active this frame...
            if stop_signal_triangle_training.status == STARTED:
                # update params
                pass
            
            # if stop_signal_triangle_training is stopping this frame...
            if stop_signal_triangle_training.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > stop_signal_triangle_training.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    stop_signal_triangle_training.tStop = t  # not accounting for scr refresh
                    stop_signal_triangle_training.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'stop_signal_triangle_training.stopped')
                    # update status
                    stop_signal_triangle_training.status = FINISHED
                    stop_signal_triangle_training.setAutoDraw(False)
            
            # *key_resp_training* updates
            waitOnFlip = False
            
            # if key_resp_training is starting this frame...
            if key_resp_training.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_training.frameNStart = frameN  # exact frame index
                key_resp_training.tStart = t  # local t and not account for scr refresh
                key_resp_training.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_training, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_training.status = STARTED
                # AllowedKeys looks like a variable named `key_resp`
                if not type(key_resp) in [list, tuple, np.ndarray]:
                    if not isinstance(key_resp, str):
                        logging.error('AllowedKeys variable `key_resp` is not string- or list-like.')
                        core.quit()
                    elif not ',' in key_resp:
                        key_resp = (key_resp,)
                    else:
                        key_resp = eval(key_resp)
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_training.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_training.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_training.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_training.getKeys(keyList=list(key_resp), ignoreKeys=["escape"], waitRelease=False)
                _key_resp_training_allKeys.extend(theseKeys)
                if len(_key_resp_training_allKeys):
                    key_resp_training.keys = _key_resp_training_allKeys[-1].name  # just the last key pressed
                    key_resp_training.rt = _key_resp_training_allKeys[-1].rt
                    key_resp_training.duration = _key_resp_training_allKeys[-1].duration
                    # was this correct?
                    if (key_resp_training.keys == str(corr_resp)) or (key_resp_training.keys == corr_resp):
                        key_resp_training.corr = 1
                    else:
                        key_resp_training.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_trainingComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_training" ---
        for thisComponent in trial_trainingComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # Run 'End Routine' code from trial_probability_training
        if trial_type == 'stop_trial':
            last_ssd = ssd
            if key_resp_training.corr:
                ssd += 0.05
            else:
                if ssd >= 0.1:
                    ssd -= 0.05
                else: ssd = ssd
                
        elif trial_type == 'go_continue_trial':
            last_ssd = last_ssd
        
        # check responses
        if key_resp_training.keys in ['', [], None]:  # No response was made
            key_resp_training.keys = None
            # was no response the correct answer?!
            if str(corr_resp).lower() == 'none':
               key_resp_training.corr = 1;  # correct non-response
            else:
               key_resp_training.corr = 0;  # failed to respond (incorrectly)
        # store data for trial_loop_training (TrialHandler)
        trial_loop_training.addData('key_resp_training.keys',key_resp_training.keys)
        trial_loop_training.addData('key_resp_training.corr', key_resp_training.corr)
        if key_resp_training.keys != None:  # we had a response
            trial_loop_training.addData('key_resp_training.rt', key_resp_training.rt)
            trial_loop_training.addData('key_resp_training.duration', key_resp_training.duration)
        # the Routine "trial_training" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "blank_training" ---
        continueRoutine = True
        # update component parameters for each repeat
        # keep track of which components have finished
        blank_trainingComponents = [no_shape_2]
        for thisComponent in blank_trainingComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank_training" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *no_shape_2* updates
            
            # if no_shape_2 is starting this frame...
            if no_shape_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                no_shape_2.frameNStart = frameN  # exact frame index
                no_shape_2.tStart = t  # local t and not account for scr refresh
                no_shape_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(no_shape_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                no_shape_2.status = STARTED
                no_shape_2.setAutoDraw(True)
            
            # if no_shape_2 is active this frame...
            if no_shape_2.status == STARTED:
                # update params
                pass
            
            # if no_shape_2 is stopping this frame...
            if no_shape_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > no_shape_2.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    no_shape_2.tStop = t  # not accounting for scr refresh
                    no_shape_2.frameNStop = frameN  # exact frame index
                    # update status
                    no_shape_2.status = FINISHED
                    no_shape_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank_trainingComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank_training" ---
        for thisComponent in blank_trainingComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "feedback_accuracy_training" ---
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from update_scores_training
        if key_resp_training.corr and not early_press_resp_training.corr:
            if trial_type in ['go_trial', 'go_fast_trial', 'go_continue_trial']:
                total_score_training += 1
                thisExp.addData('total_score_training',total_score_training)
                msg = 'richtig, +1 \n Total = ' + str(total_score_training)
            else:
                total_score_training += 3
                thisExp.addData('total_score_training',total_score_training)
                msg = 'richtig, +3 \n Total = ' + str(total_score_training)  
        elif not early_press_resp_training.corr:
            if trial_type in ['go_trial', 'go_fast_trial', 'go_continue_trial']:
                total_score_training -= 1
                thisExp.addData('total_score_training',total_score_training)
                msg = 'falsch, -1 \n Total = ' + str(total_score_training)
            else:
                total_score_training -= 3
                thisExp.addData('total_score_training',total_score_training)
                msg = 'falsch, -3 \n Total = ' + str(total_score_training)  
        else:
            total_score_training = total_score_training
            thisExp.addData('total_score_training',total_score_training)
            msg = 'zu schnell, bitte warten Sie auf das Quadrat, bevor Sie drücken'
        feedback_text_training.setText(msg)
        # keep track of which components have finished
        feedback_accuracy_trainingComponents = [feedback_text_training]
        for thisComponent in feedback_accuracy_trainingComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "feedback_accuracy_training" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedback_text_training* updates
            
            # if feedback_text_training is starting this frame...
            if feedback_text_training.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedback_text_training.frameNStart = frameN  # exact frame index
                feedback_text_training.tStart = t  # local t and not account for scr refresh
                feedback_text_training.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_text_training, 'tStartRefresh')  # time at next scr refresh
                # update status
                feedback_text_training.status = STARTED
                feedback_text_training.setAutoDraw(True)
            
            # if feedback_text_training is active this frame...
            if feedback_text_training.status == STARTED:
                # update params
                pass
            
            # if feedback_text_training is stopping this frame...
            if feedback_text_training.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedback_text_training.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    feedback_text_training.tStop = t  # not accounting for scr refresh
                    feedback_text_training.frameNStop = frameN  # exact frame index
                    # update status
                    feedback_text_training.status = FINISHED
                    feedback_text_training.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback_accuracy_trainingComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback_accuracy_training" ---
        for thisComponent in feedback_accuracy_trainingComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed n_training_trials repeats of 'trial_loop_training'
    
    
    # --- Prepare to start Routine "determine_start_ssd" ---
    continueRoutine = True
    # update component parameters for each repeat
    # keep track of which components have finished
    determine_start_ssdComponents = []
    for thisComponent in determine_start_ssdComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "determine_start_ssd" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in determine_start_ssdComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "determine_start_ssd" ---
    for thisComponent in determine_start_ssdComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # Run 'End Routine' code from code_ssd
    import numpy
    
    stop_signal_time_array = numpy.array(ssd_training_list)
    start_ssd = stop_signal_time_array.mean()
    
    
    # the Routine "determine_start_ssd" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "ssd_screen" ---
    continueRoutine = True
    # update component parameters for each repeat
    # keep track of which components have finished
    ssd_screenComponents = []
    for thisComponent in ssd_screenComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "ssd_screen" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in ssd_screenComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ssd_screen" ---
    for thisComponent in ssd_screenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "ssd_screen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Start_real_task" ---
    continueRoutine = True
    # update component parameters for each repeat
    key_resp_start_main_task.keys = []
    key_resp_start_main_task.rt = []
    _key_resp_start_main_task_allKeys = []
    # keep track of which components have finished
    Start_real_taskComponents = [text_start_task, key_resp_start_main_task]
    for thisComponent in Start_real_taskComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Start_real_task" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_start_task* updates
        
        # if text_start_task is starting this frame...
        if text_start_task.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_start_task.frameNStart = frameN  # exact frame index
            text_start_task.tStart = t  # local t and not account for scr refresh
            text_start_task.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_start_task, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_start_task.status = STARTED
            text_start_task.setAutoDraw(True)
        
        # if text_start_task is active this frame...
        if text_start_task.status == STARTED:
            # update params
            pass
        
        # *key_resp_start_main_task* updates
        waitOnFlip = False
        
        # if key_resp_start_main_task is starting this frame...
        if key_resp_start_main_task.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_start_main_task.frameNStart = frameN  # exact frame index
            key_resp_start_main_task.tStart = t  # local t and not account for scr refresh
            key_resp_start_main_task.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_start_main_task, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_start_main_task.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_start_main_task.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_start_main_task.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_start_main_task.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_start_main_task.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_start_main_task_allKeys.extend(theseKeys)
            if len(_key_resp_start_main_task_allKeys):
                key_resp_start_main_task.keys = _key_resp_start_main_task_allKeys[-1].name  # just the last key pressed
                key_resp_start_main_task.rt = _key_resp_start_main_task_allKeys[-1].rt
                key_resp_start_main_task.duration = _key_resp_start_main_task_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Start_real_taskComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Start_real_task" ---
    for thisComponent in Start_real_taskComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "Start_real_task" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "self_paced_start" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('self_paced_start.started', globalClock.getTime())
    key_resp_self_paced_start.keys = []
    key_resp_self_paced_start.rt = []
    _key_resp_self_paced_start_allKeys = []
    # keep track of which components have finished
    self_paced_startComponents = [self_paced_start_text, key_resp_self_paced_start]
    for thisComponent in self_paced_startComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "self_paced_start" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *self_paced_start_text* updates
        
        # if self_paced_start_text is starting this frame...
        if self_paced_start_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            self_paced_start_text.frameNStart = frameN  # exact frame index
            self_paced_start_text.tStart = t  # local t and not account for scr refresh
            self_paced_start_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(self_paced_start_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'self_paced_start_text.started')
            # update status
            self_paced_start_text.status = STARTED
            self_paced_start_text.setAutoDraw(True)
        
        # if self_paced_start_text is active this frame...
        if self_paced_start_text.status == STARTED:
            # update params
            pass
        
        # *key_resp_self_paced_start* updates
        waitOnFlip = False
        
        # if key_resp_self_paced_start is starting this frame...
        if key_resp_self_paced_start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_self_paced_start.frameNStart = frameN  # exact frame index
            key_resp_self_paced_start.tStart = t  # local t and not account for scr refresh
            key_resp_self_paced_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_self_paced_start, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_self_paced_start.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_self_paced_start.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_self_paced_start.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_self_paced_start.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_self_paced_start.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_self_paced_start_allKeys.extend(theseKeys)
            if len(_key_resp_self_paced_start_allKeys):
                key_resp_self_paced_start.keys = _key_resp_self_paced_start_allKeys[-1].name  # just the last key pressed
                key_resp_self_paced_start.rt = _key_resp_self_paced_start_allKeys[-1].rt
                key_resp_self_paced_start.duration = _key_resp_self_paced_start_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in self_paced_startComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "self_paced_start" ---
    for thisComponent in self_paced_startComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('self_paced_start.stopped', globalClock.getTime())
    # check responses
    if key_resp_self_paced_start.keys in ['', [], None]:  # No response was made
        key_resp_self_paced_start.keys = None
    thisExp.addData('key_resp_self_paced_start.keys',key_resp_self_paced_start.keys)
    if key_resp_self_paced_start.keys != None:  # we had a response
        thisExp.addData('key_resp_self_paced_start.rt', key_resp_self_paced_start.rt)
        thisExp.addData('key_resp_self_paced_start.duration', key_resp_self_paced_start.duration)
    thisExp.nextEntry()
    # the Routine "self_paced_start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    blocks = data.TrialHandler(nReps=4.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='blocks')
    thisExp.addLoop(blocks)  # add the loop to the experiment
    thisBlock = blocks.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            globals()[paramName] = thisBlock[paramName]
    
    for thisBlock in blocks:
        currentLoop = blocks
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        
        # --- Prepare to start Routine "trial_randomization" ---
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from randomization_block
        import random
        
        if not HALF_BLOCK:
            n_trials = 120
        else:
            n_trials = 60
        sequences_length = 6
        
        # similar to Obeso paper:
        percentage_GT = 0.34
        percentage_GF = 0.34
        percentage_GC = 0.17
        percentage_ST = 0.17
        
        n_GT = int(sequences_length * percentage_GT)
        n_GF = int(sequences_length * percentage_GF)
        n_GC = int(sequences_length * percentage_GC)
        n_ST = int(sequences_length * percentage_ST)
        
        sequences_number = int(n_trials / sequences_length)
        conditions = []
        sequence = []
        sequence_without_stop = []
        following_sequences = []
        trial_names = ['go_trial', 'go_continue_trial', 'go_fast_trial', 'stop_trial']
        
        # Generate a sequence
        sequence = (
            ['go_trial'] * int(n_GT) +
            ['go_continue_trial'] * int(n_GC) +
            ['go_fast_trial'] * int(n_GF) +
            ['stop_trial'] * int(n_ST)
        )
        
        sequence_without_stop = (
            ['go_trial'] * int(n_GT) +
            ['go_continue_trial'] * int(n_GC) +
            ['go_fast_trial'] * int(n_GF)
        )
        
        # Shuffle the sequence initially
        random.shuffle(sequence)
        random.shuffle(sequence_without_stop)
        stop_trial = (['stop_trial'])
        
        following_sequences.extend(sequence_without_stop)
        following_sequences.extend(stop_trial)
        following_sequences.extend(stop_trial)
        random.shuffle(sequence_without_stop)
        following_sequences.extend(sequence_without_stop)
        
        
        random_index = random.randint(0, sequences_number - 2)
        random_index
        
        for i in range(0, sequences_number-1, 1):
            if i == random_index:
                conditions.extend(following_sequences)
                continue
            if conditions and conditions[-1] == 'stop_trial' and sequence[0] == 'stop_trial':
                while sequence[0] == 'stop_trial':
                    random.shuffle(sequence)
                conditions.extend(sequence)
                random.shuffle(sequence)
            else:
                conditions.extend(sequence)
                random.shuffle(sequence)
        
        # keep track of which components have finished
        trial_randomizationComponents = []
        for thisComponent in trial_randomizationComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_randomization" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_randomizationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_randomization" ---
        for thisComponent in trial_randomizationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # the Routine "trial_randomization" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trial_loop = data.TrialHandler(nReps=n_trials, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='trial_loop')
        thisExp.addLoop(trial_loop)  # add the loop to the experiment
        thisTrial_loop = trial_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_loop.rgb)
        if thisTrial_loop != None:
            for paramName in thisTrial_loop:
                globals()[paramName] = thisTrial_loop[paramName]
        
        for thisTrial_loop in trial_loop:
            currentLoop = trial_loop
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_loop.rgb)
            if thisTrial_loop != None:
                for paramName in thisTrial_loop:
                    globals()[paramName] = thisTrial_loop[paramName]
            
            # --- Prepare to start Routine "fixation_period" ---
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_fixation_cross
            time_fixation = np.random.uniform(0.5,1.5)
            thisExp.addData('time_fixation', time_fixation)
            
            if right_or_left_resp.keys == 'right':
                key_resp = 'right'
            elif right_or_left_resp.keys == 'left':
                key_resp = 'left'
            
            early_press_resp.keys = []
            early_press_resp.rt = []
            _early_press_resp_allKeys = []
            # keep track of which components have finished
            fixation_periodComponents = [fixation_cross, early_press_resp]
            for thisComponent in fixation_periodComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fixation_period" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *fixation_cross* updates
                
                # if fixation_cross is starting this frame...
                if fixation_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fixation_cross.frameNStart = frameN  # exact frame index
                    fixation_cross.tStart = t  # local t and not account for scr refresh
                    fixation_cross.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fixation_cross, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_cross.started')
                    # update status
                    fixation_cross.status = STARTED
                    fixation_cross.setAutoDraw(True)
                
                # if fixation_cross is active this frame...
                if fixation_cross.status == STARTED:
                    # update params
                    pass
                
                # if fixation_cross is stopping this frame...
                if fixation_cross.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fixation_cross.tStartRefresh + time_fixation-frameTolerance:
                        # keep track of stop time/frame for later
                        fixation_cross.tStop = t  # not accounting for scr refresh
                        fixation_cross.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fixation_cross.stopped')
                        # update status
                        fixation_cross.status = FINISHED
                        fixation_cross.setAutoDraw(False)
                
                # *early_press_resp* updates
                waitOnFlip = False
                
                # if early_press_resp is starting this frame...
                if early_press_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    early_press_resp.frameNStart = frameN  # exact frame index
                    early_press_resp.tStart = t  # local t and not account for scr refresh
                    early_press_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(early_press_resp, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    early_press_resp.status = STARTED
                    # AllowedKeys looks like a variable named `right_or_left`
                    if not type(right_or_left) in [list, tuple, np.ndarray]:
                        if not isinstance(right_or_left, str):
                            logging.error('AllowedKeys variable `right_or_left` is not string- or list-like.')
                            core.quit()
                        elif not ',' in right_or_left:
                            right_or_left = (right_or_left,)
                        else:
                            right_or_left = eval(right_or_left)
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(early_press_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(early_press_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if early_press_resp is stopping this frame...
                if early_press_resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > early_press_resp.tStartRefresh + time_fixation-frameTolerance:
                        # keep track of stop time/frame for later
                        early_press_resp.tStop = t  # not accounting for scr refresh
                        early_press_resp.frameNStop = frameN  # exact frame index
                        # update status
                        early_press_resp.status = FINISHED
                        early_press_resp.status = FINISHED
                if early_press_resp.status == STARTED and not waitOnFlip:
                    theseKeys = early_press_resp.getKeys(keyList=list(right_or_left), ignoreKeys=["escape"], waitRelease=False)
                    _early_press_resp_allKeys.extend(theseKeys)
                    if len(_early_press_resp_allKeys):
                        early_press_resp.keys = _early_press_resp_allKeys[-1].name  # just the last key pressed
                        early_press_resp.rt = _early_press_resp_allKeys[-1].rt
                        early_press_resp.duration = _early_press_resp_allKeys[-1].duration
                        # was this correct?
                        if (early_press_resp.keys == str(key_resp)) or (early_press_resp.keys == key_resp):
                            early_press_resp.corr = 1
                        else:
                            early_press_resp.corr = 0
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixation_periodComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixation_period" ---
            for thisComponent in fixation_periodComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # Run 'End Routine' code from code_fixation_cross
            if early_press_resp.keys == key_resp:
                early_press = True
            # check responses
            if early_press_resp.keys in ['', [], None]:  # No response was made
                early_press_resp.keys = None
                # was no response the correct answer?!
                if str(key_resp).lower() == 'none':
                   early_press_resp.corr = 1;  # correct non-response
                else:
                   early_press_resp.corr = 0;  # failed to respond (incorrectly)
            # store data for trial_loop (TrialHandler)
            trial_loop.addData('early_press_resp.keys',early_press_resp.keys)
            trial_loop.addData('early_press_resp.corr', early_press_resp.corr)
            if early_press_resp.keys != None:  # we had a response
                psychopy_outlet.push_sample(['early'])
                trial_loop.addData('early_press_resp.rt', early_press_resp.rt)
                trial_loop.addData('early_press_resp.duration', early_press_resp.duration)
            # the Routine "fixation_period" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "trial" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trial.started', globalClock.getTime())
            # Run 'Begin Routine' code from trial_probability
            # Begin Routine tab in the Code Component
            current_trial_index = trial_loop.thisN  # Get the current iteration index
            condition = conditions[current_trial_index]
            
            color_go = 'white'
            color_stop = 'white'
            
            # initialize ssd value:
            if (first_stop_trial and condition == 'stop_trial'):
                #ssd = textbox.text
                ssd = float(start_ssd)
                first_stop_trial = False
            elif (first_stop_trial and condition == 'go_continue_trial'):
                #last_ssd = textbox.text
                #last_ssd = float(last_ssd)
                last_ssd = float(start_ssd)
                trial_type = 'go_continue_trial'
                marker_go = 'GC'
                marker_stop = 'continue'
                corr_resp = key_resp
                color_go = 'white'
                color_stop = 'blue'
                stop_signal_time = last_ssd
                go_trial_rectangle_duration = stop_signal_time
                end_routine = stop_signal_time + 1
                # end_routine = stop_signal_time + 0.5
                thisExp.addData('continue_signal_time',stop_signal_time)
                thisExp.addData('trial_type',trial_type)
                thisExp.addData('corr_resp',corr_resp)
            
            
            if not first_stop_trial:
                if condition == 'go_continue_trial':
                    trial_type = 'go_continue_trial'
                    marker_go = 'GC'
                    marker_stop = 'continue'
                    corr_resp = key_resp
                    color_go = 'white'
                    color_stop = 'blue'
                    stop_signal_time = last_ssd
                    go_trial_rectangle_duration = stop_signal_time
                    end_routine = stop_signal_time + 1
                    # end_routine = stop_signal_time + 0.5
                    thisExp.addData('continue_signal_time',stop_signal_time)
                    thisExp.addData('trial_type',trial_type)
                    thisExp.addData('corr_resp',corr_resp)
                elif condition == 'stop_trial':
                    trial_type = 'stop_trial'
                    marker_go = 'GS'
                    marker_stop = 'stop'
                    corr_resp = None
                    color_go = 'white'
                    color_stop = 'white'
                    stop_signal_time = ssd
                    end_routine = stop_signal_time + 1
                    # end_routine = stop_signal_time + 0.5
                    go_trial_rectangle_duration = stop_signal_time
                    thisExp.addData('trial_type',trial_type)
                    thisExp.addData('corr_resp',corr_resp)
                    thisExp.addData('stop_signal_time',stop_signal_time)
            
            if condition == 'go_fast_trial': 
                trial_type = 'go_fast_trial'
                marker_go = 'GF'
                corr_resp = key_resp
                color_go = 'blue'
                go_trial_rectangle_duration = 1
                stop_signal_time = 3.5
                end_routine = go_trial_rectangle_duration
                thisExp.addData('trial_type',trial_type)
                thisExp.addData('corr_resp',corr_resp)
            elif condition == 'go_trial': 
                trial_type = 'go_trial'
                marker_go = 'GO'
                corr_resp = key_resp
                color_go = 'white'
                go_trial_rectangle_duration = 1
                end_routine = go_trial_rectangle_duration
                stop_signal_time = 3.5
                thisExp.addData('trial_type',trial_type)
                thisExp.addData('corr_resp',corr_resp)
            go_rectangle.setFillColor(color_go)
            go_rectangle.setPos((0, 0))
            go_rectangle.setSize((0.5, 0.5))
            go_rectangle.setOri(0.0)
            stop_signal_triangle.setFillColor(color_stop)
            stop_signal_triangle.setPos((0, 0))
            stop_signal_triangle.setSize((0.5, 0.5))
            stop_signal_triangle.setOri(0.0)
            key_resp_experiment.keys = []
            key_resp_experiment.rt = []
            _key_resp_experiment_allKeys = []
            # keep track of which components have finished
            trialComponents = [go_rectangle, stop_signal_triangle, key_resp_experiment]
            for thisComponent in trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on local clock)
                if tThisFlip > end_routine-frameTolerance:
                    continueRoutine = False
                
                # *go_rectangle* updates
                
                # if go_rectangle is starting this frame...
                if go_rectangle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    go_rectangle.frameNStart = frameN  # exact frame index
                    go_rectangle.tStart = t  # local t and not account for scr refresh
                    go_rectangle.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(go_rectangle, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'go_rectangle.started')
                    psychopy_outlet.push_sample([marker_go])
                    # update status
                    go_rectangle.status = STARTED
                    go_rectangle.setAutoDraw(True)
                
                # if go_rectangle is active this frame...
                if go_rectangle.status == STARTED:
                    # update params
                    pass
                
                # if go_rectangle is stopping this frame...
                if go_rectangle.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > go_rectangle.tStartRefresh + go_trial_rectangle_duration-frameTolerance:
                        # keep track of stop time/frame for later
                        go_rectangle.tStop = t  # not accounting for scr refresh
                        go_rectangle.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'go_rectangle.stopped')
                        # update status
                        go_rectangle.status = FINISHED
                        go_rectangle.setAutoDraw(False)
                
                # *stop_signal_triangle* updates
                
                # if stop_signal_triangle is starting this frame...
                if stop_signal_triangle.status == NOT_STARTED and tThisFlip >= stop_signal_time-frameTolerance:
                    # keep track of start time/frame for later
                    stop_signal_triangle.frameNStart = frameN  # exact frame index
                    stop_signal_triangle.tStart = t  # local t and not account for scr refresh
                    stop_signal_triangle.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(stop_signal_triangle, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'stop_signal_triangle.started')
                    psychopy_outlet.push_sample([marker_stop])
                    # update status
                    stop_signal_triangle.status = STARTED
                    stop_signal_triangle.setAutoDraw(True)
                
                # if stop_signal_triangle is active this frame...
                if stop_signal_triangle.status == STARTED:
                    # update params
                    pass
                
                # if stop_signal_triangle is stopping this frame...
                if stop_signal_triangle.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > stop_signal_triangle.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        stop_signal_triangle.tStop = t  # not accounting for scr refresh
                        stop_signal_triangle.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'stop_signal_triangle.stopped')
                        # update status
                        stop_signal_triangle.status = FINISHED
                        stop_signal_triangle.setAutoDraw(False)
                
                # *key_resp_experiment* updates
                waitOnFlip = False
                
                # if key_resp_experiment is starting this frame...
                if key_resp_experiment.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_experiment.frameNStart = frameN  # exact frame index
                    key_resp_experiment.tStart = t  # local t and not account for scr refresh
                    key_resp_experiment.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_experiment, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_experiment.started')
                    # update status
                    key_resp_experiment.status = STARTED
                    # AllowedKeys looks like a variable named `key_resp`
                    if not type(key_resp) in [list, tuple, np.ndarray]:
                        if not isinstance(key_resp, str):
                            logging.error('AllowedKeys variable `key_resp` is not string- or list-like.')
                            core.quit()
                        elif not ',' in key_resp:
                            key_resp = (key_resp,)
                        else:
                            key_resp = eval(key_resp)
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_experiment.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_experiment.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_experiment.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_experiment.getKeys(keyList=list(key_resp), ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_experiment_allKeys.extend(theseKeys)
                    if len(_key_resp_experiment_allKeys):
                        psychopy_outlet.push_sample(['resp'])
                        key_resp_experiment.keys = _key_resp_experiment_allKeys[-1].name  # just the last key pressed
                        key_resp_experiment.rt = _key_resp_experiment_allKeys[-1].rt
                        key_resp_experiment.duration = _key_resp_experiment_allKeys[-1].duration
                        # was this correct?
                        if (key_resp_experiment.keys == str(corr_resp)) or (key_resp_experiment.keys == corr_resp):
                            key_resp_experiment.corr = 1
                        else:
                            key_resp_experiment.corr = 0
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trial.stopped', globalClock.getTime())
            # Run 'End Routine' code from trial_probability
            if trial_type == 'stop_trial':
                last_ssd = ssd
                if key_resp_experiment.corr:
                    ssd += 0.05
                else:
                    if ssd >= 0.1:
                        ssd -= 0.05
                    else: ssd = ssd
            elif trial_type == 'go_continue_trial':
                last_ssd = last_ssd
            
            
            
            # check responses
            if key_resp_experiment.keys in ['', [], None]:  # No response was made
                key_resp_experiment.keys = None
                # was no response the correct answer?!
                if str(corr_resp).lower() == 'none':
                   key_resp_experiment.corr = 1;  # correct non-response
                else:
                   key_resp_experiment.corr = 0;  # failed to respond (incorrectly)
            # store data for trial_loop (TrialHandler)
            trial_loop.addData('key_resp_experiment.keys',key_resp_experiment.keys)
            trial_loop.addData('key_resp_experiment.corr', key_resp_experiment.corr)
            if key_resp_experiment.keys != None:  # we had a response
                trial_loop.addData('key_resp_experiment.rt', key_resp_experiment.rt)
                trial_loop.addData('key_resp_experiment.duration', key_resp_experiment.duration)
            # the Routine "trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "blank" ---
            continueRoutine = True
            # update component parameters for each repeat
            late_key_resp1.keys = []
            late_key_resp1.rt = []
            _late_key_resp1_allKeys = []
            # keep track of which components have finished
            blankComponents = [no_shape, late_key_resp1]
            for thisComponent in blankComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "blank" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on local clock)
                if tThisFlip > 0.5-frameTolerance:
                    continueRoutine = False
                
                # *no_shape* updates
                
                # if no_shape is starting this frame...
                if no_shape.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    no_shape.frameNStart = frameN  # exact frame index
                    no_shape.tStart = t  # local t and not account for scr refresh
                    no_shape.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(no_shape, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    no_shape.status = STARTED
                    no_shape.setAutoDraw(True)
                
                # if no_shape is active this frame...
                if no_shape.status == STARTED:
                    # update params
                    pass
                
                # if no_shape is stopping this frame...
                if no_shape.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > no_shape.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        no_shape.tStop = t  # not accounting for scr refresh
                        no_shape.frameNStop = frameN  # exact frame index
                        # update status
                        no_shape.status = FINISHED
                        no_shape.setAutoDraw(False)
                
                # *late_key_resp1* updates
                waitOnFlip = False
                
                # if late_key_resp1 is starting this frame...
                if late_key_resp1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    late_key_resp1.frameNStart = frameN  # exact frame index
                    late_key_resp1.tStart = t  # local t and not account for scr refresh
                    late_key_resp1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(late_key_resp1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'late_key_resp1.started')
                    # update status
                    late_key_resp1.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(late_key_resp1.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(late_key_resp1.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if late_key_resp1.status == STARTED and not waitOnFlip:
                    theseKeys = late_key_resp1.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                    _late_key_resp1_allKeys.extend(theseKeys)
                    if len(_late_key_resp1_allKeys):
                        late_key_resp1.keys = _late_key_resp1_allKeys[-1].name  # just the last key pressed
                        late_key_resp1.rt = _late_key_resp1_allKeys[-1].rt
                        late_key_resp1.duration = _late_key_resp1_allKeys[-1].duration
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in blankComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "blank" ---
            for thisComponent in blankComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # check responses
            if late_key_resp1.keys in ['', [], None]:  # No response was made
                late_key_resp1.keys = None
            trial_loop.addData('late_key_resp1.keys',late_key_resp1.keys)
            if late_key_resp1.keys != None:  # we had a response
                psychopy_outlet.push_sample(['late'])
                trial_loop.addData('late_key_resp1.rt', late_key_resp1.rt)
                trial_loop.addData('late_key_resp1.duration', late_key_resp1.duration)
            # the Routine "blank" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "feedback_accuracy" ---
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from update_scores
            if key_resp_experiment.corr and not early_press_resp.corr:
                if trial_type in ['go_trial', 'go_fast_trial', 'go_continue_trial']:
                    total_score += 1
                    thisExp.addData('total_score',total_score)
                    msg = 'richtig, +1 \n Total = ' + str(total_score)
                else:
                    total_score += 3
                    total_nb_success_stop += 1
                    thisExp.addData('total_score',total_score)
                    thisExp.addData('total_nb_success_stop', total_nb_success_stop)
                    msg = 'richtig, +3 \n Total = ' + str(total_score)  
            elif not early_press_resp.corr:
                if trial_type in ['go_trial', 'go_fast_trial', 'go_continue_trial']:
                    total_score -= 1
                    thisExp.addData('total_score',total_score)
                    msg = 'falsch, -1 \n Total = ' + str(total_score)
                else:
                    total_score -= 3
                    total_nb_unsuccess_stop += 1
                    thisExp.addData('total_nb_unsuccess_stop',total_nb_unsuccess_stop)
                    thisExp.addData('total_score',total_score)
                    msg = 'falsch, -3 \n Total = ' + str(total_score)  
            else:
                total_score = total_score
                thisExp.addData('total_score',total_score)
                msg = 'zu schnell, bitte warten Sie auf das Quadrat, bevor Sie drücken'
            feedback_text.setText(msg)
            late_key_resp2.keys = []
            late_key_resp2.rt = []
            _late_key_resp2_allKeys = []
            # keep track of which components have finished
            feedback_accuracyComponents = [feedback_text, late_key_resp2]
            for thisComponent in feedback_accuracyComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "feedback_accuracy" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on local clock)
                if tThisFlip > 1-frameTolerance:
                    continueRoutine = False
                
                # *feedback_text* updates
                
                # if feedback_text is starting this frame...
                if feedback_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    feedback_text.frameNStart = frameN  # exact frame index
                    feedback_text.tStart = t  # local t and not account for scr refresh
                    feedback_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(feedback_text, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    feedback_text.status = STARTED
                    feedback_text.setAutoDraw(True)
                
                # if feedback_text is active this frame...
                if feedback_text.status == STARTED:
                    # update params
                    pass
                
                # if feedback_text is stopping this frame...
                if feedback_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > feedback_text.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        feedback_text.tStop = t  # not accounting for scr refresh
                        feedback_text.frameNStop = frameN  # exact frame index
                        # update status
                        feedback_text.status = FINISHED
                        feedback_text.setAutoDraw(False)
                
                # *late_key_resp2* updates
                waitOnFlip = False
                
                # if late_key_resp2 is starting this frame...
                if late_key_resp2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    late_key_resp2.frameNStart = frameN  # exact frame index
                    late_key_resp2.tStart = t  # local t and not account for scr refresh
                    late_key_resp2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(late_key_resp2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'late_key_resp2.started')
                    # update status
                    late_key_resp2.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(late_key_resp2.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(late_key_resp2.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if late_key_resp2.status == STARTED and not waitOnFlip:
                    theseKeys = late_key_resp2.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                    _late_key_resp2_allKeys.extend(theseKeys)
                    if len(_late_key_resp2_allKeys):
                        late_key_resp2.keys = _late_key_resp2_allKeys[-1].name  # just the last key pressed
                        late_key_resp2.rt = _late_key_resp2_allKeys[-1].rt
                        late_key_resp2.duration = _late_key_resp2_allKeys[-1].duration
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in feedback_accuracyComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "feedback_accuracy" ---
            for thisComponent in feedback_accuracyComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # check responses
            if late_key_resp2.keys in ['', [], None]:  # No response was made
                late_key_resp2.keys = None
            trial_loop.addData('late_key_resp2.keys',late_key_resp2.keys)
            if late_key_resp2.keys != None:  # we had a response
                psychopy_outlet.push_sample(['late'])
                trial_loop.addData('late_key_resp2.rt', late_key_resp2.rt)
                trial_loop.addData('late_key_resp2.duration', late_key_resp2.duration)
            # the Routine "feedback_accuracy" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed n_trials repeats of 'trial_loop'
        
        
        # --- Prepare to start Routine "find_nb_each_stop_trials" ---
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from sum_success_unsuccess
        if total_nb_success_stop != 0:
            nb_success_stop = total_nb_success_stop
        else:
            nb_success_stop = 0
            
        if total_nb_unsuccess_stop != 0:
            nb_unsuccess_stop = total_nb_unsuccess_stop
        else:
            nb_unsuccess_stop = 0
        
        msg_stops = ('number of successful stop trials= ' 
        + str(nb_success_stop) 
        + ', number of unsuccessful stop trials= ' 
        + str(nb_unsuccess_stop))
        # keep track of which components have finished
        find_nb_each_stop_trialsComponents = []
        for thisComponent in find_nb_each_stop_trialsComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "find_nb_each_stop_trials" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in find_nb_each_stop_trialsComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "find_nb_each_stop_trials" ---
        for thisComponent in find_nb_each_stop_trialsComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # Run 'End Routine' code from sum_success_unsuccess
        if (blocks.thisN >= 2 and nb_success_stop >= 25 and nb_unsuccess_stop >= 25):
            blocks.finished = True
            break_rep=0
        if (blocks.thisN >= 2 and nb_success_stop < 25 and nb_unsuccess_stop < 25):
             HALF_BLOCK=True
        if blocks.thisN >= 3:
            break_rep=0
        
        # the Routine "find_nb_each_stop_trials" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        break_screen_loop = data.TrialHandler(nReps=break_rep, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='break_screen_loop')
        thisExp.addLoop(break_screen_loop)  # add the loop to the experiment
        thisBreak_screen_loop = break_screen_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisBreak_screen_loop.rgb)
        if thisBreak_screen_loop != None:
            for paramName in thisBreak_screen_loop:
                globals()[paramName] = thisBreak_screen_loop[paramName]
        
        for thisBreak_screen_loop in break_screen_loop:
            currentLoop = break_screen_loop
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisBreak_screen_loop.rgb)
            if thisBreak_screen_loop != None:
                for paramName in thisBreak_screen_loop:
                    globals()[paramName] = thisBreak_screen_loop[paramName]
            
            # --- Prepare to start Routine "break_between_block" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('break_between_block.started', globalClock.getTime())
            break1_resp.keys = []
            break1_resp.rt = []
            _break1_resp_allKeys = []
            # keep track of which components have finished
            break_between_blockComponents = [break1_text, break1_resp]
            for thisComponent in break_between_blockComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "break_between_block" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *break1_text* updates
                
                # if break1_text is starting this frame...
                if break1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    break1_text.frameNStart = frameN  # exact frame index
                    break1_text.tStart = t  # local t and not account for scr refresh
                    break1_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(break1_text, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    break1_text.status = STARTED
                    break1_text.setAutoDraw(True)
                
                # if break1_text is active this frame...
                if break1_text.status == STARTED:
                    # update params
                    pass
                
                # *break1_resp* updates
                waitOnFlip = False
                
                # if break1_resp is starting this frame...
                if break1_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    break1_resp.frameNStart = frameN  # exact frame index
                    break1_resp.tStart = t  # local t and not account for scr refresh
                    break1_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(break1_resp, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    break1_resp.status = STARTED
                    # AllowedKeys looks like a variable named `key_resp`
                    if not type(key_resp) in [list, tuple, np.ndarray]:
                        if not isinstance(key_resp, str):
                            logging.error('AllowedKeys variable `key_resp` is not string- or list-like.')
                            core.quit()
                        elif not ',' in key_resp:
                            key_resp = (key_resp,)
                        else:
                            key_resp = eval(key_resp)
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(break1_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(break1_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if break1_resp.status == STARTED and not waitOnFlip:
                    theseKeys = break1_resp.getKeys(keyList=list(key_resp), ignoreKeys=["escape"], waitRelease=False)
                    _break1_resp_allKeys.extend(theseKeys)
                    if len(_break1_resp_allKeys):
                        break1_resp.keys = _break1_resp_allKeys[-1].name  # just the last key pressed
                        break1_resp.rt = _break1_resp_allKeys[-1].rt
                        break1_resp.duration = _break1_resp_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in break_between_blockComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "break_between_block" ---
            for thisComponent in break_between_blockComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('break_between_block.stopped', globalClock.getTime())
            # the Routine "break_between_block" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
        # completed break_rep repeats of 'break_screen_loop'
        
    # completed 4.0 repeats of 'blocks'
    
    
    # --- Prepare to start Routine "end_screen" ---
    continueRoutine = True
    # update component parameters for each repeat
    end_resp.keys = []
    end_resp.rt = []
    _end_resp_allKeys = []
    # keep track of which components have finished
    end_screenComponents = [end_text, end_resp]
    for thisComponent in end_screenComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end_screen" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_text* updates
        
        # if end_text is starting this frame...
        if end_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_text.frameNStart = frameN  # exact frame index
            end_text.tStart = t  # local t and not account for scr refresh
            end_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_text, 'tStartRefresh')  # time at next scr refresh
            # update status
            end_text.status = STARTED
            end_text.setAutoDraw(True)
        
        # if end_text is active this frame...
        if end_text.status == STARTED:
            # update params
            pass
        
        # *end_resp* updates
        waitOnFlip = False
        
        # if end_resp is starting this frame...
        if end_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_resp.frameNStart = frameN  # exact frame index
            end_resp.tStart = t  # local t and not account for scr refresh
            end_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_resp, 'tStartRefresh')  # time at next scr refresh
            # update status
            end_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(end_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(end_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if end_resp.status == STARTED and not waitOnFlip:
            theseKeys = end_resp.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
            _end_resp_allKeys.extend(theseKeys)
            if len(_end_resp_allKeys):
                end_resp.keys = _end_resp_allKeys[-1].name  # just the last key pressed
                end_resp.rt = _end_resp_allKeys[-1].rt
                end_resp.duration = _end_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end_screenComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_screen" ---
    for thisComponent in end_screenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "end_screen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
