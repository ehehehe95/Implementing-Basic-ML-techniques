package com.yooong.studynotefragment.recyclerView;

import android.content.Context;
import android.content.Intent;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.google.android.material.button.MaterialButton;
import com.yooong.studynotefragment.ProblemOpened;
import com.yooong.studynotefragment.R;

import java.io.File;
import java.util.ArrayList;


public class ImageAdapter extends RecyclerView.Adapter<ImageAdapter.ImageHolder> {

    private ArrayList<String> imageUriList;

    private ArrayList<MaterialButton> deleteButtonList;
    public ImageAdapter(ArrayList<String> imageUriList){
        this.imageUriList=imageUriList;
        this.deleteButtonList=new ArrayList<>();
        Log.d("imageAdapter","initiated imageUriList");
    }

//    public ImageAdapter(ArrayList<String> imageUriList, ArrayList<String> solutionImageList, ArrayList<String> solutionTextList) {
//        this.imageUriList = imageUriList;
//        this.solutionImageList=solutionImageList;
//        this.solutionTextList=solutionTextList;
//        deleteButtonList=new ArrayList<>();
//    }

    public static class ImageHolder extends RecyclerView.ViewHolder{
        public ImageView imageView;
        public MaterialButton deleteButton;
        public ImageHolder(View itemView) {
            super(itemView);
            imageView = itemView.findViewById(R.id.problem);
            deleteButton = itemView.findViewById(R.id.deleteImageButton);

        }
    }

    @Override
    public ImageAdapter.ImageHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        Log.d("imageAdapter","onCreateViewHolder");
        Context context = parent.getContext();
        LayoutInflater inflater = (LayoutInflater) context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);

        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.image_view, parent, false) ;
        ImageAdapter.ImageHolder vh = new ImageAdapter.ImageHolder(view);

        return vh;
    }


    @Override
    public void onBindViewHolder(@NonNull ImageAdapter.ImageHolder holder, int position) {
        Log.d("imageAdapter","OnBindViewHolderStart");
        String imageUri = imageUriList.get(position);
        Log.d("imageAdapter",imageUri.toString());
        Glide.with(holder.itemView.getContext()).load(imageUri).into(holder.imageView);
        Context mContext=holder.imageView.getContext();
        deleteButtonList.add(holder.deleteButton);

        holder.imageView.setOnLongClickListener(new View.OnLongClickListener() {
            @Override
            public boolean onLongClick(View v) {
                for(MaterialButton button : deleteButtonList){
                    button.setVisibility(View.VISIBLE);
                }
                return true;
            }
        });

        holder.imageView.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(mContext, ProblemOpened.class);
                intent.putExtra("imageUriList",imageUriList);
                intent.putExtra("index",position);

                for(MaterialButton button : deleteButtonList){
                    button.setVisibility(View.INVISIBLE);
                }

                mContext.startActivity(intent);

            }
        });
        holder.deleteButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d("test8","delete files"+imageUriList.get(position));
                String deleteLocation = imageUriList.get(position);
                File deleteImage = new File(deleteLocation);
                File deleteSolution = new File(deleteLocation.replace(".jpg","_solution.jpg"));

                Log.d("test8","deleteFOlder"+deleteImage.getAbsolutePath());
                Log.d("test8","deleteSolution"+deleteSolution.getAbsolutePath());

                try{
                    deleteSolution.delete();
                }catch(Exception e){
                    Log.d("test8","sth wrong deleting solution");
                }
                deleteImage.delete();
                imageUriList.remove(position);
                deleteButtonList.remove(position);
                notifyDataSetChanged();
            }
        });
    }

    /**
     * Returns the total number of items in the data set held by the adapter.
     *
     * @return The total number of items in this adapter.
     */
    @Override
    public int getItemCount() {
        return imageUriList.size();
    }
}
