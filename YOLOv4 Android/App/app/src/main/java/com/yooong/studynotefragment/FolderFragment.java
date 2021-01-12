package com.yooong.studynotefragment;

import android.content.Context;
import android.content.DialogInterface;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.material.dialog.MaterialAlertDialogBuilder;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.yooong.studynotefragment.recyclerView.folderAdapter;

import java.io.File;
import java.util.ArrayList;

import static com.yooong.studynotefragment.MainActivity.dir;
import static com.yooong.studynotefragment.MainActivity.fragmentPage;

public class FolderFragment extends Fragment {

    private RecyclerView recyclerView;
    private RecyclerView.Adapter mAdapter;
    private RecyclerView.LayoutManager layoutManager;
    private ArrayList<String> fileNames;

    FloatingActionButton gallery,camera,add_button;


    public interface folderClickListener{
        void folderClicked(String folderName);
        void makeFolder(String folderName, int subjectId);
    }

    folderClickListener folderClickListener;


    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //get Lists of the folders inside studyNote

        File[] files = dir.listFiles();
        fileNames = new ArrayList<>();

        for(File f : files){
            Log.d("makeStudyNote",f.getName());
            fileNames.add(f.getName());
        }

    }

    @Override
    public void onAttach(Context context) {
        super.onAttach(context);
        try {
            folderClickListener = (folderClickListener) context;
        } catch (ClassCastException e) {
            throw new ClassCastException(context.toString() + " must implement onSomeEventListener");
        }

        camera=getActivity().findViewById(R.id.addFromCamera);
        gallery=getActivity().findViewById(R.id.addFromGallery);
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_folder,container,false);

        recyclerView = (RecyclerView) view.findViewById(R.id.folderList);
        recyclerView.setHasFixedSize(true);

        // use a grid layout manager
        layoutManager = new GridLayoutManager(getContext(),3);
        recyclerView.setLayoutManager(layoutManager);

        // specify an adapter (see also next example)
        mAdapter = new folderAdapter(this.getActivity(),fileNames,dir.getAbsolutePath(),new folderAdapter.OnFolderClickListener(){

            @Override
            public void onFolderClicked(String folderName) {
                Log.d("fromFolderFragment",folderName);
                folderClickListener.folderClicked(folderName);

            }

            @Override
            public void onDeleteButtonClicked(String folderName) {
                Log.d("test6","from fragment");
                Log.d("test6",dir.getAbsolutePath());

                //TODO should ask whether to delete
                new MaterialAlertDialogBuilder(getContext())
                        .setTitle("폴더 삭제")
                        .setMessage(folderName+"을 삭제하시겠습니까?")
                        .setPositiveButton("OK", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialog, int which) {
                                Log.d("test6","폴더 삭제 시작");
                                File deleteFolder = new File(dir,folderName);
                                String[] children = deleteFolder.list();
                                for (int i = 0; i < children.length; i++) {
                                    new File(deleteFolder, children[i]).delete();
                                }

                                deleteFolder.delete();
                                fileNames.remove(folderName);
                                mAdapter.notifyDataSetChanged();
                            }
                        })
                        .setNegativeButton("NO",null)
                        .show();

//                File deleteFolder = new File(dir,folderName);
//                String[] children = deleteFolder.list();
//                for (int i = 0; i < children.length; i++) {
//                    new File(deleteFolder, children[i]).delete();
//                }
//
//                deleteFolder.delete();
//                fileNames.remove(folderName);
//                mAdapter.notifyDataSetChanged();

            }
        });
        recyclerView.setAdapter(mAdapter);
        return view;
    }

    public void makeFolder(String folderName,int subjectId){
        fileNames.add(folderName);
        mAdapter.notifyDataSetChanged();

    }


    //to handle floating button use on resume

    @Override
    public void onResume() {
        super.onResume();
        fragmentPage=0;
        camera.setVisibility(View.GONE);
        gallery.setVisibility(View.GONE);
        Log.d("test10","Resumed!!!!!!!!");
    }
}