import mongoose from 'mongoose'

const connectDB=async()=>{
    
    mongoose.connection.on("connected",()=>{
        console.log("DB Connected");
    })
    try{
    await mongoose.connect(`${process.env.MONGODB_URI}/hackathon`);
    }catch(error){
        console.log("MONGODB_URL:", process.env.MONGODB_URI);
        console.log("Failed to connect",error.message);
    }
}
export default connectDB;