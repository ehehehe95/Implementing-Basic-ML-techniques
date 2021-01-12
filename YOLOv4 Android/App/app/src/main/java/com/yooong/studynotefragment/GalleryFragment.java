package com.yooong.studynotefragment;

import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.yooong.studynotefragment.recyclerView.ImageAdapter;

import java.util.ArrayList;

import static com.yooong.studynotefragment.MainActivity.fragmentPage;
import static com.yooong.studynotefragment.MainActivity.problemNote;


public class GalleryFragment extends Fragment {
    private RecyclerView recyclerView;
    private RecyclerView.Adapter mAdapter;
    private RecyclerView.LayoutManager layoutManager;
    private ArrayList<String> imageUriList;
    private TextView folderNameView;


    GalleryFragment(){
        super();
    }

    GalleryFragment(ArrayList<String> imageUriList){
        this.imageUriList = imageUriList;
        Log.d("galleryFragment","imageUriList initiated");
    }


    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

    }


    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        Log.d("galleryFragment","onCreateView start");
        View view = inflater.inflate(R.layout.fragment_gallery,container,false);
        //
        folderNameView = view.findViewById(R.id.folderTitle);
        folderNameView.setText(problemNote.getName());

        // make grid view
        recyclerView = (RecyclerView) view.findViewById(R.id.imageList);
        recyclerView.setHasFixedSize(true);

        // use a grid layout manager
        layoutManager = new GridLayoutManager(getContext(),2);
        recyclerView.setLayoutManager(layoutManager);
        Log.d("galleryFragment","recyclerView complete");
        // specify an adapter (see also next example)
        mAdapter = new ImageAdapter(imageUriList);
        recyclerView.setAdapter(mAdapter);
        Log.d("galleryFragment","number of item inside : "+mAdapter.getItemCount());
        return view;

    }

    public void addedImage(String destinationFile) {
        Log.d("uriTest",destinationFile);
        imageUriList.add(destinationFile);
        mAdapter.notifyDataSetChanged();
    }

    @Override
    public void onResume() {
        super.onResume();
        fragmentPage=1;
        Log.d("test10","Gallery Resumed!!!!!!!!");
    }
}