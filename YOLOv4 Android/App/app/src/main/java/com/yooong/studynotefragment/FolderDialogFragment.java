package com.yooong.studynotefragment;

import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.fragment.app.DialogFragment;

import java.util.HashMap;

public class FolderDialogFragment extends DialogFragment {
    private EditText editText;
    private RadioGroup subjectGroup;
    public dialogListener dialogListener;

    private TextView actionOk, actionCancel;

    public int iconId;
    private RadioButton ic_folder, ic_history, ic_cal, ic_law, ic_science, ic_writing, ic_symbols, ic_pet, ic_school, ic_idea, ic_art, ic_heart;
    HashMap<Integer,RadioButton> iconMap = new HashMap<>();

    public interface dialogListener {
        void sendFolderData(String folderName, int subjectId);

    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {


        View view = inflater.inflate(R.layout.fragment_folder_dialog, container, false);

        editText = view.findViewById(R.id.folderName);

        actionCancel = view.findViewById(R.id.action_cancel);
        actionOk = view.findViewById(R.id.action_ok);
        subjectGroup = view.findViewById(R.id.subjectGroup);

        //radio Button

        iconId = R.id.ic_folder;

        ic_folder = view.findViewById(R.id.ic_folder);
        ic_history = view.findViewById(R.id.ic_history);
        ic_cal = view.findViewById(R.id.ic_cal);
        ic_law = view.findViewById(R.id.ic_law);
        ic_science = view.findViewById(R.id.ic_science);
        ic_writing = view.findViewById(R.id.ic_writing);
        ic_symbols = view.findViewById(R.id.ic_symbols);
        ic_pet = view.findViewById(R.id.ic_pet);
        ic_school = view.findViewById(R.id.ic_school);
        ic_idea = view.findViewById(R.id.ic_idea);
        ic_art = view.findViewById(R.id.ic_art);
        ic_heart = view.findViewById(R.id.ic_heart);

        iconMap.put(R.id.ic_folder,ic_folder);
        iconMap.put(R.id.ic_history,ic_history);
        iconMap.put(R.id.ic_cal,ic_cal);
        iconMap.put(R.id.ic_law,ic_law);
        iconMap.put(R.id.ic_science,ic_science);
        iconMap.put(R.id.ic_writing,ic_writing);
        iconMap.put(R.id.ic_symbols,ic_symbols);
        iconMap.put(R.id.ic_pet,ic_pet);
        iconMap.put(R.id.ic_school,ic_school);
        iconMap.put(R.id.ic_idea,ic_idea);
        iconMap.put(R.id.ic_art,ic_art);
        iconMap.put(R.id.ic_heart,ic_heart);


        makeRadioClickListener(ic_folder);
        makeRadioClickListener(ic_history);
        makeRadioClickListener(ic_cal);
        makeRadioClickListener(ic_law);
        makeRadioClickListener(ic_science);
        makeRadioClickListener(ic_writing);
        makeRadioClickListener(ic_symbols);
        makeRadioClickListener(ic_pet);
        makeRadioClickListener(ic_school);
        makeRadioClickListener(ic_idea);
        makeRadioClickListener(ic_art);
        makeRadioClickListener(ic_heart);


        actionCancel.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                getDialog().dismiss();
            }
        });


        actionOk.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                String folderName = editText.getText().toString();

                if(iconId==-1){
                    iconId=R.id.ic_folder;
                }

                if (folderName.equals("폴더이름을 입력하세요!") || folderName.equals("")) {
                    Toast.makeText(getActivity(), "폴더이름을 입력하세요!", Toast.LENGTH_SHORT).show();
                    dismiss();
                }
                else{
                dialogListener.sendFolderData(folderName, iconId);
                getDialog().dismiss();
                }
            }
        });
        return view;
    }

    @Override
    public void onAttach(Context context) {
        super.onAttach(context);
        try {
            dialogListener = (dialogListener) context;
        } catch (ClassCastException e) {
            Log.e("test", "onAttach: ClassCastException: " + e.getMessage());
        }
    }

    private void makeRadioClickListener(RadioButton radioButton) {
        radioButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d("test5","thisButtonId"+radioButton.getId());
                Log.d("test5","iconId start : "+iconId);
                if (iconId == radioButton.getId()) {
                    Log.d("test5","seletedSameItem, should change it to false");

                    iconId = -1;
                    radioButton.setChecked(false);
                } else {
                    if (iconId != -1) {
                        Log.d("test5","already selected icon, change it to none checked");
                        Log.d("test5","already selected button id "+iconMap.get(iconId));
                        RadioButton checkedButton = iconMap.get(iconId);
                        checkedButton.setChecked(false);
                    }
                    Log.d("test5","change this to checked");
                    radioButton.setChecked(true);
                    iconId=radioButton.getId();
                }
                Log.d("test5","iconId end : "+iconId);
            }
        });
    }
}