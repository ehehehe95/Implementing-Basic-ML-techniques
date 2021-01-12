package com.yooong.studynotefragment;

import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.RadioGroup;
import android.widget.Toast;

import androidx.fragment.app.DialogFragment;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.slider.RangeSlider;

import java.util.ArrayList;
import java.util.List;

import static com.yooong.studynotefragment.MainActivity.MINIMUM_CONFIDENCE_TF_OD_API;
import static com.yooong.studynotefragment.MainActivity.onlyWrong;

public class AiDialogFragment extends DialogFragment {

    private RadioGroup radioGroup;
    private RangeSlider aiSlider;
    private MaterialButton aiOkButton, aiCancelButton;

    private int problemModeId;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_ai_dialog, container, false);
        radioGroup = view.findViewById(R.id.problemConfig);
        aiSlider = view.findViewById(R.id.aiSensitivitySlider);
        aiOkButton = view.findViewById(R.id.ai_config_ok);
        aiCancelButton = view.findViewById(R.id.ai_config_cancel);
        List<Float> value = new ArrayList<>();
        value.add(0, MINIMUM_CONFIDENCE_TF_OD_API * 100);
        aiSlider.setValues(value);
        if(onlyWrong){
            radioGroup.check(R.id.onlyWrong);
        }else{
            radioGroup.check(R.id.allProblem);
        }

        aiCancelButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                getDialog().dismiss();
            }
        });

        aiOkButton.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                problemModeId = radioGroup.getCheckedRadioButtonId();
                if (problemModeId == R.id.allProblem) {
                    Log.d("test12","모든문제모드");
                    onlyWrong = false;
                } else {
                    Log.d("test12","틀린문제 모드");
                    onlyWrong = true;
                }

                MINIMUM_CONFIDENCE_TF_OD_API = aiSlider.getValues().get(0)/100;
                Log.d("test12", "finalValue" + aiSlider.getValues());

                Toast toast =
                        Toast.makeText(
                                getActivity(), "변경 완료", Toast.LENGTH_SHORT);
                toast.show();

                getDialog().dismiss();
            }
        });

        return view;
    }
}